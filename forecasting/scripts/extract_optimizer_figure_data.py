#!/usr/bin/env python3
"""Build the compact public optimizer-figure dataset from approved private histories.

The optimizer figures (convergence, sampling matrix, surrogate landscape) are the
only consumers of the historical optimizer searches, and they read just one file
per optimizer family: ``<search_dir>/<optimizer>/optimizer_results.csv``. The
multi-gigabyte per-trial run artifacts are never touched. This script projects
those 32 approved histories into a compact, sanitised public dataset that
reproduces all 24 figures.

Source selection is *not* rediscovered here. It is read from the approved audit
matrix, and every source file is checked against the checksum recorded by that
audit before a single row is copied.

Values are copied as raw strings via the ``csv`` module: categorical tokens
(``full``, ``linear_quadratic``) and booleans (``true``/``false``) must survive
byte-for-byte, because the plotting code orders categories by ``sorted()`` over
the raw values and parses booleans itself. Nothing is reformatted, rounded, or
normalised.

The output carries no generation timestamp, so a re-extraction from the same
sources is byte-identical.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

_SCRIPT = Path(__file__).resolve()
_REPO_ROOT = _SCRIPT.parents[2]
_FORECASTING_SRC = _REPO_ROOT / "forecasting" / "src"
for _bootstrap in (_REPO_ROOT, _FORECASTING_SRC):
    if str(_bootstrap) not in sys.path:
        sys.path.insert(0, str(_bootstrap))

from common.logging_setup import get_logger  # noqa: E402

logger = get_logger("forecasting.scripts.extract_optimizer_figure_data")

DATASET_VERSION = "1.0.0"
PRIVATE_SOURCE_COMMIT = "7d574a44b40f27e40302e9946ef6af5243222cab"
PUBLIC_BASE_COMMIT = "4c6370b1267ae28bad7dbcf8fd31638720ba7494"
AUDIT_VERSION = "optimizer-authoritative-source-audit-v001"

OPTIMIZERS: tuple[str, ...] = ("dummy", "forest", "gbrt", "gp")
OPTIMIZER_DISPLAY = {
    "dummy": "Random",
    "forest": "Random forest",
    "gbrt": "GBRT",
    "gp": "Gaussian process",
}
CASES: tuple[str, ...] = (
    "ngrc_full64", "ngrc_latent8", "rc_full64", "rc_latent8",
    "perc_full64", "perc_latent8", "pinn_full64", "pinn_latent8",
)
EXPECTED_ROWS_PER_FAMILY = {
    "ngrc_full64": 120, "ngrc_latent8": 140, "rc_full64": 80, "rc_latent8": 80,
    "perc_full64": 80, "perc_latent8": 100, "pinn_full64": 80, "pinn_latent8": 80,
}

METRIC = "valid_relative_l2_horizon_time"
OBJECTIVE = "maximize"

# Columns identifying the row and its provenance, written before the parameters.
CORE_COLUMNS: tuple[str, ...] = (
    "case_id", "model", "representation", "optimizer",
    "source_run_id", "source_search_order", "search_stage",
    "trial", "status",
)

# Never published: private filesystem paths, run identifiers that embed them, a
# raw error field, and objective_value (a sign-flipped mirror of the metric).
DROPPED_COLUMNS: frozenset[str] = frozenset(
    {"output_dir", "summary_path", "candidate_run_id", "run_id", "error", "objective_value"}
)

# Any of these appearing in a public dataset file is a hard failure.
FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "/home/", "/scratch/", "/tmp/", "DataDrivenModelsOfChaoticPDEs", "dsnf120h",
    "output_dir", "summary_path",
)

CANONICAL_CONFIG_DIR = Path("final_thesis_package_v001/configs/03_plots")
CONFIG_FOR_MODEL = {
    "NGRC": "optimizer_search_ngrc.json",
    "RC": "optimizer_search_rc.json",
    "PERC": "optimizer_search_perc.json",
    "PINN": "optimizer_search_pinn.json",
}


class ExtractionError(RuntimeError):
    """Raised when the extraction cannot proceed safely."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def required_parameters(repo_root: Path, model: str) -> list[str]:
    """Config-declared parameter names, in the canonical declared order."""
    config_path = repo_root / CANONICAL_CONFIG_DIR / CONFIG_FOR_MODEL[model]
    if not config_path.exists():
        raise ExtractionError(f"Canonical config missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return [str(p["name"]) for p in config["search_plots"]["parameters"]]


def load_source_matrix(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise ExtractionError(f"Approved source matrix not found: {path}")

    rows = read_csv_rows(path)

    if len(rows) != 32:
        raise ExtractionError(f"Approved matrix must hold exactly 32 rows, found {len(rows)}.")

    for row in rows:
        if row.get("confidence") != "HIGH":
            raise ExtractionError(f"Row not at HIGH confidence: {row.get('case_id')}/{row.get('optimizer_internal')}")
        if row.get("approved_for_future_extraction") != "YES":
            raise ExtractionError(f"Row not approved: {row.get('case_id')}/{row.get('optimizer_internal')}")

    by_case: dict[str, list[str]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row["optimizer_internal"])

    if sorted(by_case) != sorted(CASES):
        raise ExtractionError(f"Unexpected case set: {sorted(by_case)}")

    for case, families in by_case.items():
        if sorted(families) != list(OPTIMIZERS):
            raise ExtractionError(f"Case {case} does not hold exactly {OPTIMIZERS}: {sorted(families)}")

    return rows


def load_audit_checksums(path: Path) -> dict[str, str]:
    """Map absolute source path -> SHA-256 recorded by the approved audit."""
    if not path.exists():
        raise ExtractionError(f"Audit inventory not found: {path}")
    return {
        row["exact_private_path"]: row["file_sha256"]
        for row in read_csv_rows(path)
        if row.get("likely_authoritative") == "YES"
    }


def scan_forbidden(text: str, label: str) -> None:
    for needle in FORBIDDEN_SUBSTRINGS:
        if needle in text:
            raise ExtractionError(f"Refusing to publish {label}: contains {needle!r}.")


def extract_history(
    entry: dict[str, str], repo_root: Path, private_repo: Path, checksums: dict[str, str]
) -> tuple[list[str], list[dict[str, str]], dict[str, Any]]:
    """Project one approved private history into the public column set."""
    case_id = entry["case_id"]
    optimizer = entry["optimizer_internal"]
    source = Path(entry["authoritative_private_csv"])

    if not source.exists():
        raise ExtractionError(f"Approved source missing: {source}")

    try:
        source.relative_to(private_repo)
    except ValueError as exc:
        raise ExtractionError(f"Source lies outside the private repository: {source}") from exc

    # The audit recorded a checksum for every approved history; drift means the
    # source changed since approval and the selection is no longer trustworthy.
    expected = checksums.get(str(source))
    if expected is None:
        raise ExtractionError(f"No approved checksum recorded for {source}")
    actual = sha256_file(source)
    if actual != expected:
        raise ExtractionError(f"Source checksum drift for {source}: {actual} != {expected}")

    # Guard against the five known top-level PINN duplicate/aggregate files: an
    # authoritative history always sits in a directory named for its optimizer.
    if source.parent.name != optimizer:
        raise ExtractionError(
            f"Source is not at loader depth <search_dir>/{optimizer}/optimizer_results.csv: {source}"
        )

    params = required_parameters(repo_root, entry["model"])
    rows = read_csv_rows(source)
    present = set(rows[0].keys()) if rows else set()

    missing = [p for p in params if p not in present]
    if missing:
        raise ExtractionError(f"{case_id}/{optimizer} missing config-declared parameters: {missing}")
    if METRIC not in present:
        raise ExtractionError(f"{case_id}/{optimizer} missing metric column {METRIC}.")

    for row in rows:
        row_optimizer = str(row.get("optimizer", optimizer)).strip()
        if row_optimizer and row_optimizer != optimizer:
            raise ExtractionError(
                f"{case_id}: row optimizer {row_optimizer!r} disagrees with directory {optimizer!r}."
            )

    columns = list(CORE_COLUMNS) + params + [METRIC]
    out_rows: list[dict[str, str]] = []

    for row in rows:
        record = {
            "case_id": case_id,
            "model": entry["model"],
            "representation": entry["representation"],
            "optimizer": optimizer,
            "source_run_id": entry["run_name"],
            "source_search_order": entry["source_directory_order"],
            "search_stage": entry["search_stage"],
            "trial": row["trial"],
            "status": row.get("status", "ok"),
        }
        for name in params:
            record[name] = row[name]
        record[METRIC] = row[METRIC]
        out_rows.append(record)

    stats = {
        "source_row_count": len(rows),
        "extracted_row_count": len(out_rows),
        "source_sha256": actual,
        "required_parameter_count": len(params),
        "parameters": params,
    }
    return columns, out_rows, stats


def write_dataset_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" plus an explicit "\n" terminator keeps the bytes identical
    # across platforms, so the manifest checksums are stable.
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    scan_forbidden(path.read_text(encoding="utf-8"), str(path))


def build(
    *, private_repo: Path, source_matrix: Path, audit_inventory: Path, output_root: Path,
    repo_root: Path, overwrite: bool,
) -> dict[str, Any]:
    if output_root.exists():
        if not overwrite:
            raise ExtractionError(
                f"Output root already exists and --overwrite was not given: {output_root}"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    entries = load_source_matrix(source_matrix)
    checksums = load_audit_checksums(audit_inventory)

    manifest_rows: list[dict[str, Any]] = []
    per_case_params: dict[str, list[str]] = {}
    total_rows = 0

    for entry in sorted(entries, key=lambda e: (e["case_id"], e["optimizer_internal"])):
        case_id, optimizer = entry["case_id"], entry["optimizer_internal"]
        columns, rows, stats = extract_history(entry, repo_root, private_repo, checksums)

        expected_rows = EXPECTED_ROWS_PER_FAMILY[case_id]
        if len(rows) != expected_rows:
            raise ExtractionError(
                f"{case_id}/{optimizer}: expected {expected_rows} rows, extracted {len(rows)}."
            )

        rel = Path(case_id) / optimizer / "optimizer_results.csv"
        target = output_root / rel
        write_dataset_csv(target, columns, rows)
        per_case_params[case_id] = stats["parameters"]
        total_rows += len(rows)

        manifest_rows.append({
            "case_id": case_id,
            "model": entry["model"],
            "representation": entry["representation"],
            "optimizer": optimizer,
            "source_run_id": entry["run_name"],
            "source_search_order": entry["source_directory_order"],
            "search_stage": entry["search_stage"],
            "source_relative_path": str(Path(entry["authoritative_private_csv"]).relative_to(private_repo)),
            "source_sha256": stats["source_sha256"],
            "source_row_count": stats["source_row_count"],
            "extracted_relative_path": str(rel),
            "extracted_sha256": sha256_file(target),
            "extracted_row_count": stats["extracted_row_count"],
            "required_parameter_count": stats["required_parameter_count"],
            "canonical_config_relative_path": str(CANONICAL_CONFIG_DIR / CONFIG_FOR_MODEL[entry["model"]]),
        })
        logger.info("Extracted | case=%s optimizer=%s rows=%d", case_id, optimizer, len(rows))

    if len(manifest_rows) != 32:
        raise ExtractionError(f"Expected 32 extracted histories, wrote {len(manifest_rows)}.")
    if total_rows != 3040:
        raise ExtractionError(f"Expected 3040 total trials, wrote {total_rows}.")

    _write_metadata(output_root, manifest_rows, per_case_params, total_rows)
    _write_checksum_manifest(output_root)
    logger.info("Dataset complete | histories=%d trials=%d root=%s", len(manifest_rows), total_rows, output_root)
    return {"histories": len(manifest_rows), "trials": total_rows}


def _write_metadata(
    output_root: Path, manifest_rows: list[dict[str, Any]],
    per_case_params: dict[str, list[str]], total_rows: int,
) -> None:
    fields = list(manifest_rows[0].keys())
    with (output_root / "source_selection_manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)

    schema = {
        "dataset_version": DATASET_VERSION,
        "description": "Compact optimizer trial histories reproducing the 24 optimizer figures.",
        "cases": list(CASES),
        "optimizer_vocabulary": {k: OPTIMIZER_DISPLAY[k] for k in OPTIMIZERS},
        "core_columns": list(CORE_COLUMNS),
        "metric": METRIC,
        "objective": OBJECTIVE,
        "primary_key": ["case_id", "optimizer", "trial"],
        "status_filter_rule": "Rows are plotted only when status == 'ok' and the metric parses as finite.",
        "best_row_rule": (
            "The search-time best row is derived at plot time: the row maximising "
            f"{METRIC}. It is not stored, and it is not necessarily the final published model."
        ),
        "raw_value_policy": (
            "Categorical and boolean tokens are stored exactly as recorded by the search "
            "(for example 'full', 'linear_quadratic', 'true', 'false'). Plot-time display "
            "mapping is applied by the plotting code and never written into the data."
        ),
        "column_types": {
            "case_id": "string", "model": "string", "representation": "string",
            "optimizer": "string (one of the optimizer vocabulary)",
            "source_run_id": "string", "source_search_order": "integer",
            "search_stage": "string", "trial": "integer (unique per case+optimizer)",
            "status": "string", METRIC: "float",
        },
        "parameter_columns_by_case": per_case_params,
        "total_trials": total_rows,
        "trials_per_case": {c: EXPECTED_ROWS_PER_FAMILY[c] * len(OPTIMIZERS) for c in CASES},
    }
    (output_root / "schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    provenance = {
        "dataset_version": DATASET_VERSION,
        "private_source_repository_commit": PRIVATE_SOURCE_COMMIT,
        "public_base_commit": PUBLIC_BASE_COMMIT,
        "plotting_worktree_uncommitted": True,
        "approved_audit_version": AUDIT_VERSION,
        "source_matrix_filename": "authoritative_optimizer_source_matrix_v001.csv",
        "extraction_script": "forecasting/scripts/extract_optimizer_figure_data.py",
        "canonical_configs": sorted(
            str(CANONICAL_CONFIG_DIR / name) for name in set(CONFIG_FOR_MODEL.values())
        ),
        "histories": [
            {
                "case_id": r["case_id"], "optimizer": r["optimizer"],
                "source_run_id": r["source_run_id"],
                "source_search_order": r["source_search_order"],
                "source_relative_path": r["source_relative_path"],
                "source_sha256": r["source_sha256"],
                "extracted_relative_path": r["extracted_relative_path"],
                "extracted_sha256": r["extracted_sha256"],
                "row_count": r["extracted_row_count"],
            }
            for r in manifest_rows
        ],
        "notes": (
            "Source paths are recorded relative to the private repository root. The large "
            "per-trial artifacts of each search remain private because no optimizer figure "
            "reads them."
        ),
    }
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "README.md").write_text(_readme(total_rows), encoding="utf-8")

    for name in ("schema.json", "provenance.json", "README.md", "source_selection_manifest.csv"):
        scan_forbidden((output_root / name).read_text(encoding="utf-8"), name)


def _write_checksum_manifest(output_root: Path) -> None:
    manifest_name = "dataset_sha256_manifest.csv"
    files = sorted(
        p for p in output_root.rglob("*") if p.is_file() and p.name != manifest_name
    )
    rows = [
        {
            "relative_path": str(p.relative_to(output_root)),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
        for p in files
    ]
    with (output_root / manifest_name).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["relative_path", "size_bytes", "sha256"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        target = output_root / row["relative_path"]
        if sha256_file(target) != row["sha256"]:
            raise ExtractionError(f"Checksum verification failed for {row['relative_path']}.")
    logger.info("Checksum manifest verified | files=%d", len(rows))


def _readme(total_rows: int) -> str:
    return f"""# Compact optimizer-search dataset

This directory holds the optimizer trial histories required to reproduce the
twenty-four optimizer figures of the thesis: eight convergence plots, eight
sampling-matrix plots and eight surrogate-landscape plots.

## Purpose

The optimizer figures are the only artifacts that consume the hyperparameter
search histories, and they read a single table per optimizer family. Publishing
those tables makes the figures reproducible from this repository alone. The
per-trial run artifacts of the original searches (model checkpoints, predicted
trajectories, intermediate arrays) amount to tens of gigabytes, are never read by
any figure, and therefore remain private.

## Contents

Eight forecasting cases, each searched with four optimizer families:

| directory | internal name | display name |
|---|---|---|
| `dummy/` | `dummy` | Random |
| `forest/` | `forest` | Random forest |
| `gbrt/` | `gbrt` | GBRT |
| `gp/` | `gp` | Gaussian process |

The cases are `ngrc_full64`, `ngrc_latent8`, `rc_full64`, `rc_latent8`,
`perc_full64`, `perc_latent8`, `pinn_full64` and `pinn_latent8`. Together the
thirty-two tables hold {total_rows} successful optimizer trials.

```
<case_id>/<optimizer>/optimizer_results.csv
```

Alongside the tables: `schema.json` (column contract), `provenance.json` (source
lineage and checksums), `source_selection_manifest.csv` (one row per approved
history) and `dataset_sha256_manifest.csv` (checksums for every file here).

## Columns

`case_id`, `model` and `representation` identify the forecasting case.
`optimizer` names the search family and always agrees with the containing
directory. `source_run_id`, `source_search_order` and `search_stage` preserve the
original search campaign, which matters for the two cases whose families were
partitioned across more than one search directory: `ngrc_latent8` (random and
Gaussian-process families in one run, tree-based families in another) and
`pinn_full64` (one directory per family). `trial` is the original trial number and
is unique per case and optimizer, not globally. `status` is retained so the
plotting filter remains meaningful. `valid_relative_l2_horizon_time` is the
validation-horizon metric that every optimizer figure maximises. The remaining
columns are the search-space parameters declared by the canonical plotting
configuration for that model family.

Values are stored exactly as the searches recorded them. Categorical tokens such
as `full` and `linear_quadratic`, and booleans such as `true` and `false`, are
never normalised: the plotting code orders categories by sorting raw values and
applies its own display mapping. Rewriting them here would silently change the
axis ordering of the published figures.

## Reproducing the figures

With the pinned environment (`requirements.txt` / `environment-lock.yml`;
`scikit-learn==1.3.2` in particular, which governs the surrogate fit):

```
python forecasting/scripts/plot_forecasting_visuals.py \\
  --config forecasting/configs/03_plots/optimizer_search_ngrc_public.json \\
  --output-root <preview directory>
```

and likewise for the `rc`, `perc` and `pinn` public configurations. The
surrogate landscape refits a Gaussian process at plot time; because every case
holds fewer rows than the configured `max_train_points`, no subsampling occurs
and the fit is deterministic for a given environment.

## Interpreting the marked optimum

> The red star in optimizer sampling-matrix and surrogate-landscape figures
> denotes the search-time metric optimum. It is the final selected model for the
> two NGRC cases. For the six seed-reranked or deterministically repaired
> forecasting cases, it is diagnostic and does not necessarily denote the final
> published configuration.

The two NGRC cases are deterministic, so the search argmax was adopted directly.
The six remaining cases were finalised differently: `rc_full64`, `rc_latent8`,
`perc_full64`, `perc_latent8` and `pinn_latent8` by re-ranking the top candidates
over fifteen fresh seeds and selecting the highest median validation horizon, and
`pinn_full64` by a deterministic CPU-controlled repair followed by a staged
re-rank. For those cases the optimizer figures remain valid as search
diagnostics, but they are not selection evidence; the seed-robustness figures
serve that role.

No selected-candidate column is shipped. Doing so would conflate the search-time
optimum, which the plotting code derives from the metric, with the final
post-search selection, which was made by a different procedure.

## Provenance and reproducibility

The trial histories originate from the private research repository at commit
`{PRIVATE_SOURCE_COMMIT}`, projected by
`forecasting/scripts/extract_optimizer_figure_data.py` against the approved
source-selection audit. `provenance.json` records, for every history, the source
path relative to the private repository root together with the source and
extracted checksums.

This dataset supports scientific reproduction (the same trials, extrema, optima
and surfaces) and visual reproduction (the same figures). Byte-identical PNG
checksums across different machines are not guaranteed: Gaussian-process
hyperparameter fitting and font rasterisation both vary at the last digits with
the numerical libraries and platform in use.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the compact public optimizer-figure dataset from approved private histories."
    )
    parser.add_argument("--private-repo", required=True, type=Path, help="Private repository root (read-only).")
    parser.add_argument("--source-matrix", required=True, type=Path, help="Approved authoritative source matrix CSV.")
    parser.add_argument(
        "--audit-inventory", type=Path, default=None,
        help="Approved candidate inventory CSV holding source checksums "
             "(default: candidate_search_inventory_v001.csv beside the matrix).",
    )
    parser.add_argument("--output-root", required=True, type=Path, help="Public dataset root to create.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output root.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inventory = args.audit_inventory or args.source_matrix.parent / "candidate_search_inventory_v001.csv"
    try:
        result = build(
            private_repo=args.private_repo.resolve(),
            source_matrix=args.source_matrix.resolve(),
            audit_inventory=inventory.resolve(),
            output_root=args.output_root.resolve(),
            repo_root=_REPO_ROOT,
            overwrite=args.overwrite,
        )
    except ExtractionError as exc:
        logger.error("Extraction refused | %s", exc)
        return 2
    logger.info("Done | histories=%d trials=%d", result["histories"], result["trials"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
