"""Contract tests for the compact public optimizer-figure dataset.

These tests read only the published dataset under
``forecasting/data/optimizer_search`` plus the public configs. They load no
private data, train nothing, and run no optimizer search.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "forecasting" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA_ROOT = _REPO_ROOT / "forecasting" / "data" / "optimizer_search"
PUBLIC_CONFIG_DIR = _REPO_ROOT / "forecasting" / "configs" / "03_plots"
CANONICAL_CONFIG_DIR = _REPO_ROOT / "final_thesis_package_v001" / "configs" / "03_plots"

OPTIMIZERS = ("dummy", "forest", "gbrt", "gp")
METRIC = "valid_relative_l2_horizon_time"
CASES = (
    "ngrc_full64", "ngrc_latent8", "rc_full64", "rc_latent8",
    "perc_full64", "perc_latent8", "pinn_full64", "pinn_latent8",
)
# From the approved authoritative source audit (rows per optimizer family).
EXPECTED_PER_FAMILY = {
    "ngrc_full64": 120, "ngrc_latent8": 140, "rc_full64": 80, "rc_latent8": 80,
    "perc_full64": 80, "perc_latent8": 100, "pinn_full64": 80, "pinn_latent8": 80,
}
EXPECTED_TOTAL = 3040
# Search-time argmax recorded by the audit: (optimizer, trial).
EXPECTED_ARGMAX = {
    "ngrc_full64": ("forest", "108"), "ngrc_latent8": ("dummy", "60"),
    "rc_full64": ("gp", "78"), "rc_latent8": ("gbrt", "63"),
    "perc_full64": ("forest", "42"), "perc_latent8": ("forest", "47"),
    # pinn_full64 now holds the v003 physics-active search; its search-time argmax is GBRT
    # trial 22 at 31.2. That is the search optimum only -- the published configuration is
    # candidate C5 (dummy trial 47), chosen on 15-seed median validation horizon.
    "pinn_full64": ("gbrt", "22"), "pinn_latent8": ("gp", "53"),
}
DISALLOWED_COLUMNS = {"output_dir", "summary_path", "candidate_run_id", "run_id", "error", "objective_value"}
FORBIDDEN_TEXT = ("/home/", "/scratch/", "/tmp/", "DataDrivenModelsOfChaoticPDEs", "dsnf120h")
CONFIG_FOR_MODEL = {"NGRC": "ngrc", "RC": "rc", "PERC": "perc", "PINN": "pinn"}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _all_trial_csvs() -> list[Path]:
    # Explicit case/optimizer walk, never a recursive glob: a recursive glob
    # would also pick up top-level aggregate files if any were ever added.
    return [
        DATA_ROOT / case / opt / "optimizer_results.csv"
        for case in CASES
        for opt in OPTIMIZERS
    ]


class TestDatasetStructure(unittest.TestCase):
    def test_dataset_exists(self):
        self.assertTrue(DATA_ROOT.is_dir(), f"dataset missing: {DATA_ROOT}")

    def test_exactly_eight_cases(self):
        found = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
        self.assertEqual(found, sorted(CASES))

    def test_each_case_has_four_optimizer_dirs(self):
        for case in CASES:
            found = sorted(p.name for p in (DATA_ROOT / case).iterdir() if p.is_dir())
            self.assertEqual(found, sorted(OPTIMIZERS), f"case {case}")

    def test_exactly_32_trial_csvs(self):
        paths = _all_trial_csvs()
        self.assertEqual(len(paths), 32)
        for p in paths:
            self.assertTrue(p.is_file(), f"missing {p}")


class TestDatasetContent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.by_case = defaultdict(list)
        cls.by_file = {}
        for case in CASES:
            for opt in OPTIMIZERS:
                rows = _rows(DATA_ROOT / case / opt / "optimizer_results.csv")
                cls.by_file[(case, opt)] = rows
                cls.by_case[case].extend(rows)

    def test_total_row_count(self):
        self.assertEqual(sum(len(v) for v in self.by_case.values()), EXPECTED_TOTAL)

    def test_per_case_and_per_family_counts(self):
        for case in CASES:
            per = EXPECTED_PER_FAMILY[case]
            for opt in OPTIMIZERS:
                self.assertEqual(len(self.by_file[(case, opt)]), per, f"{case}/{opt}")
            self.assertEqual(len(self.by_case[case]), per * 4, case)

    def test_all_rows_status_ok(self):
        for case, rows in self.by_case.items():
            self.assertEqual({r["status"] for r in rows}, {"ok"}, case)

    def test_metric_present_and_finite(self):
        for case, rows in self.by_case.items():
            for r in rows:
                value = float(r[METRIC])
                self.assertTrue(math.isfinite(value), f"{case} non-finite metric")

    def test_config_declared_parameters_present(self):
        for case in CASES:
            model = self.by_case[case][0]["model"]
            config = json.loads(
                (CANONICAL_CONFIG_DIR / f"optimizer_search_{CONFIG_FOR_MODEL[model]}.json").read_text()
            )
            declared = [p["name"] for p in config["search_plots"]["parameters"]]
            columns = set(self.by_case[case][0].keys())
            for name in declared:
                self.assertIn(name, columns, f"{case} missing {name}")

    def test_primary_key_unique(self):
        keys = Counter(
            (r["case_id"], r["optimizer"], r["trial"])
            for rows in self.by_case.values() for r in rows
        )
        self.assertEqual([k for k, n in keys.items() if n > 1], [])
        self.assertEqual(len(keys), EXPECTED_TOTAL)

    def test_optimizer_column_matches_directory(self):
        for (case, opt), rows in self.by_file.items():
            self.assertEqual({r["optimizer"] for r in rows}, {opt}, f"{case}/{opt}")

    def test_case_id_matches_directory(self):
        for (case, opt), rows in self.by_file.items():
            self.assertEqual({r["case_id"] for r in rows}, {case})

    def test_no_disallowed_columns(self):
        for (case, opt), rows in self.by_file.items():
            self.assertEqual(set(rows[0]) & DISALLOWED_COLUMNS, set(), f"{case}/{opt}")

    def test_no_selected_candidate_field(self):
        # The best row is derived at plot time; storing a selection would
        # conflate the search-time optimum with the final published model.
        for rows in self.by_file.values():
            for column in rows[0]:
                self.assertNotIn("selected", column.lower())

    def test_raw_categorical_values_unnormalized(self):
        ngrc = self.by_case["ngrc_full64"]
        self.assertEqual({r["feature_set"] for r in ngrc}, {"full", "linear_quadratic"})
        self.assertTrue({r["include_bias"] for r in ngrc} <= {"True", "False", "true", "false"})
        for r in ngrc:
            for key in ("feature_set", "include_bias"):
                self.assertNotIn("\n", r[key])
                self.assertNotIn("Full library", r[key])

    def test_search_time_argmax_matches_audit(self):
        for case, (opt, trial) in EXPECTED_ARGMAX.items():
            best = max(self.by_case[case], key=lambda r: float(r[METRIC]))
            self.assertEqual((best["optimizer"], best["trial"]), (opt, trial), case)


class TestSanitization(unittest.TestCase):
    def test_no_private_paths_anywhere(self):
        for path in sorted(DATA_ROOT.rglob("*")):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in FORBIDDEN_TEXT:
                self.assertNotIn(needle, text, f"{path.name} leaks {needle!r}")


class TestManifests(unittest.TestCase):
    def test_dataset_checksums_validate(self):
        manifest = DATA_ROOT / "dataset_sha256_manifest.csv"
        rows = _rows(manifest)
        self.assertGreaterEqual(len(rows), 36)
        for row in rows:
            target = DATA_ROOT / row["relative_path"]
            self.assertTrue(target.is_file(), row["relative_path"])
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            self.assertEqual(digest, row["sha256"], row["relative_path"])

    def test_source_selection_manifest_has_32_rows(self):
        rows = _rows(DATA_ROOT / "source_selection_manifest.csv")
        self.assertEqual(len(rows), 32)
        self.assertEqual(sorted({r["case_id"] for r in rows}), sorted(CASES))

    def test_provenance_records_private_commit_and_uncommitted_flag(self):
        prov = json.loads((DATA_ROOT / "provenance.json").read_text())
        self.assertEqual(
            prov["private_source_repository_commit"],
            "7d574a44b40f27e40302e9946ef6af5243222cab",
        )
        self.assertTrue(prov["plotting_worktree_uncommitted"])
        self.assertEqual(len(prov["histories"]), 32)

    def test_schema_declares_key_and_metric(self):
        schema = json.loads((DATA_ROOT / "schema.json").read_text())
        self.assertEqual(schema["primary_key"], ["case_id", "optimizer", "trial"])
        self.assertEqual(schema["metric"], METRIC)
        self.assertEqual(schema["objective"], "maximize")
        self.assertEqual(schema["total_trials"], EXPECTED_TOTAL)

    def test_readme_discloses_star_meaning(self):
        # Normalise wrapping and blockquote markers so the assertion tracks the
        # prose rather than the markdown layout.
        raw = (DATA_ROOT / "README.md").read_text()
        readme = " ".join(
            line.lstrip().lstrip(">").strip() for line in raw.splitlines()
        )
        readme = " ".join(readme.split())
        self.assertIn("search-time metric optimum", readme)
        self.assertIn("does not necessarily denote the final published configuration", readme)
        self.assertIn("It is the final selected model for the two NGRC cases", readme)


class TestPublicConfigs(unittest.TestCase):
    CONFIGS = (
        "optimizer_search_ngrc_public.json",
        "optimizer_search_rc_public.json",
        "optimizer_search_perc_public.json",
        "optimizer_search_pinn_public.json",
    )

    def test_configs_exist(self):
        for name in self.CONFIGS:
            self.assertTrue((PUBLIC_CONFIG_DIR / name).is_file(), name)

    def test_configs_point_only_at_public_dataset(self):
        for name in self.CONFIGS:
            cfg = json.loads((PUBLIC_CONFIG_DIR / name).read_text())
            for group in cfg["inputs"]["optimizer_groups"]:
                for d in group["search_dirs"]:
                    self.assertTrue(
                        d.startswith("forecasting/data/optimizer_search/"),
                        f"{name} references {d}",
                    )
                    self.assertTrue((_REPO_ROOT / d).is_dir(), d)

    def test_configs_do_not_write_into_frozen_package(self):
        for name in self.CONFIGS:
            cfg = json.loads((PUBLIC_CONFIG_DIR / name).read_text())
            self.assertNotIn("final_thesis_package_v001", cfg["output"]["base_dir"])

    def test_scientific_settings_preserved_from_canonical(self):
        for name in self.CONFIGS:
            model = name.replace("optimizer_search_", "").replace("_public.json", "")
            pub = json.loads((PUBLIC_CONFIG_DIR / name).read_text())["search_plots"]
            canon = json.loads(
                (CANONICAL_CONFIG_DIR / f"optimizer_search_{model}.json").read_text()
            )["search_plots"]
            for key in ("metric", "objective", "parameters", "surrogate", "use_lyapunov", "lambda_max"):
                self.assertEqual(pub.get(key), canon.get(key), f"{name}:{key}")


class TestLoaderIntegration(unittest.TestCase):
    def test_loader_reads_all_histories_without_duplicates(self):
        from forecasting.visualization.io import load_optimizer_results

        total = 0
        for case in CASES:
            rows = load_optimizer_results([DATA_ROOT / case])
            self.assertEqual(len(rows), EXPECTED_PER_FAMILY[case] * 4, case)
            self.assertEqual(sorted({r["optimizer"] for r in rows}), sorted(OPTIMIZERS), case)
            total += len(rows)
        self.assertEqual(total, EXPECTED_TOTAL)

    def test_loader_best_row_matches_audit(self):
        from forecasting.visualization.io import load_optimizer_results
        from forecasting.visualization.search_plots import _best_row, ok_rows

        for case, (opt, trial) in EXPECTED_ARGMAX.items():
            rows = ok_rows(load_optimizer_results([DATA_ROOT / case]), METRIC)
            best = _best_row(rows, METRIC, "maximize")
            self.assertEqual((best["optimizer"], best["trial"]), (opt, trial), case)


class TestSurrogateRendering(unittest.TestCase):
    """End-to-end render of all eight surrogate landscapes from public data only.

    Guards the colorbar-offset correction at the level that matters: every case
    must render, and none may leave a detached offset string on the colorbar.
    """

    def test_all_eight_surrogates_render_without_colorbar_offset(self):
        import tempfile

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from forecasting.config import load_config
        from forecasting.visualization import search_plots as SP
        from forecasting.visualization.io import load_optimizer_results, resolve_path

        offsets: dict[str, str] = {}
        rendered: list[str] = []

        with tempfile.TemporaryDirectory() as tmp:
            for family in ("ngrc", "rc", "perc", "pinn"):
                cfg = load_config(
                    _REPO_ROOT / "forecasting/configs/03_plots"
                    / f"optimizer_search_{family}_public.json"
                )
                search_cfg = cfg["search_plots"]
                specs = SP.build_parameter_specs(search_cfg["parameters"])
                surrogate = search_cfg.get("surrogate", {})

                for group in cfg["inputs"]["optimizer_groups"]:
                    label = group["label"]
                    dirs = [resolve_path(d, _REPO_ROOT) for d in group["search_dirs"]]
                    for path in dirs:
                        self.assertTrue(
                            str(path).startswith(str(_REPO_ROOT / "forecasting/data")),
                            f"{label} reads outside the public dataset: {path}",
                        )

                    rows = SP.ok_rows(load_optimizer_results(dirs), METRIC)
                    captured: dict = {}
                    original = SP.save_figure

                    def capture(fig, *a, _c=captured, _o=original, **kw):
                        _c["axes"] = list(fig.axes)
                        return _o(fig, *a, **kw)

                    SP.save_figure = capture
                    try:
                        written = SP.plot_surrogate_landscape(
                            rows,
                            specs=specs,
                            metric=METRIC,
                            objective=search_cfg.get("objective", "maximize"),
                            output_base=Path(tmp) / label / "surrogate",
                            formats=["png"],
                            dpi=72,
                            max_train_points=int(surrogate.get("max_train_points", 1200)),
                            random_seed=int(surrogate.get("random_seed", 0)),
                            use_lyapunov=bool(search_cfg.get("use_lyapunov", False)),
                            lambda_max=float(search_cfg.get("lambda_max", 1.0)),
                        )
                    finally:
                        SP.save_figure = original

                    self.assertEqual(len(written), 1, label)
                    rendered.append(label)

                    # Identify the colourbar by its role -- a tall, very narrow
                    # axes -- rather than by a literal x position. Pinning the
                    # coordinate made this test fail the moment the colourbar
                    # moved left to stop its label overflowing the printed page,
                    # even though what the test actually checks (the offset text)
                    # was unaffected.
                    cax = next(
                        ax for ax in captured["axes"]
                        if ax.get_position().width < 0.05
                        and ax.get_position().height > 0.5
                    )
                    offsets[label] = cax.yaxis.get_offset_text().get_text()
                    plt.close("all")

        self.assertEqual(sorted(rendered), sorted(CASES))
        for label, text in offsets.items():
            self.assertEqual(text, "", f"{label} still shows colorbar offset {text!r}")


if __name__ == "__main__":
    unittest.main()
