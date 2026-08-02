#!/usr/bin/env python3
"""Reproduce and validate the final thesis results.

    python scripts/reproduce_final_thesis.py                       # interactive menu
    python scripts/reproduce_final_thesis.py --list
    python scripts/reproduce_final_thesis.py --plan --results ngrc_full64
    python scripts/reproduce_final_thesis.py --validate-existing --yes
    python scripts/reproduce_final_thesis.py --results ngrc_full64 rc_latent8 --yes

Every run writes a self-contained, timestamped folder under reproduction_runs/ holding
only the requested material.

Three invariants this script enforces and never lets the caller override:

  1. final_thesis_package_v001/ is READ-ONLY. The package configs write their outputs
     back into the package (output.base_dir / output_dir / output_root all point there),
     so a config is never handed to a runner as-is: it is redirected into the run folder
     first. See redirect_config().
  2. The KSE solver is never re-run and latent6_trial002 is never retrained. Both are
     frozen; a chaotic trajectory cannot be regenerated bit-identically, and the latent6
     retrain was attempted and rejected. They are checksum-validated only.
  3. Tolerances come from repro_validation/tolerances/tolerance_policy.json. Two are
     flagged review_required there; this script refuses to silently pass them. See
     resolve_tolerance() and compare().

Expected values and tolerances are the Stage 9 contract under repro_validation/. This
script reads that contract; it does not restate it.
"""

from __future__ import annotations

import argparse
import csv
import functools
import datetime as _dt
import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.logging_setup import (  # noqa: E402  (repo-root bootstrap must precede this)
    add_logging_arguments,
    configure_from_args,
    get_logger,
    get_presentation,
)

PACKAGE = REPO_ROOT / "final_thesis_package_v001"
CONTRACT = REPO_ROOT / "repro_validation"
RUNS_ROOT = REPO_ROOT / "reproduction_runs"

TARGETS_CSV = CONTRACT / "expected" / "validation_targets.csv"
POLICY_JSON = CONTRACT / "tolerances" / "tolerance_policy.json"
TABLE_MANIFEST = CONTRACT / "manifests" / "package_table_sha256.csv"

# Interpreters. The split is not cosmetic: full-64 forecasting is pure numpy/scipy and
# was validated under the system interpreter, while anything that decodes latent ->
# physical needs torch, which lived only in the torch (originally conda) interpreter.
#
# SYS_PY / CONDA_PY are the ACTIVE interpreters used to launch child processes. They
# default to the original HPC paths so behaviour on the original machine is unchanged,
# and are re-resolved at startup by resolve_interpreters() so the same code runs from a
# portable virtual environment. A single interpreter (for example one project .venv with
# torch installed) may satisfy both roles. Override with --system-python / --torch-python
# or the REPRO_SYSTEM_PYTHON / REPRO_TORCH_PYTHON environment variables.
_HPC_SYS_PY = "/usr/bin/python3"
_HPC_CONDA_PY = "/apps/python/3.12-conda/envs/pytorch2.6-py3.12/bin/python"
SYS_PY = _HPC_SYS_PY
CONDA_PY = _HPC_CONDA_PY

# The H-03 deterministic profile (scripts/repro_env.sh), applied to child processes.
# Single-threaded BLAS is load-bearing: ngrc_full64's validation horizon moves 0.29%
# between pinned and unpinned runs, which at 379.9 is ~1.10 -- ten times the runtime
# tolerance.
PINNED_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "MPLBACKEND": "Agg",
}

PASS, FAIL, REVIEW, SKIPPED, ERROR = "PASS", "FAIL", "REVIEW_REQUIRED", "SKIPPED", "ERROR"

RULE = "─" * 40

# Structured operational/diagnostic logging (component REPRO / VALIDATE on stderr for
# warnings and errors); `pres` is the plain, unprefixed stdout channel for the menu,
# --list/--plan text, tables and the final summaries. Neither uses a bare print.
logger = get_logger("repro")
val_logger = get_logger("repro.validate")
pres = get_presentation()


# ───────────────────────────── result registry ─────────────────────────────


@dataclass
class Result:
    result_id: str
    display_name: str
    group: str  # 00_simulation | 01_ae_sindy | 02_forecasting
    mode: str  # validate-only | reproduce | checkpoint-reanalysis
    difficulty: str  # fast | medium | medium/heavy | heavy
    status: str  # frozen | reproduce | validate-only
    action: str  # implemented | implemented_gpu_gated | implemented_validate_only
    target_ids: list[str]
    package_result_path: str
    expected_source: str
    package_config_path: str | None = None
    plots: list[str] = field(default_factory=list)
    laptop_safe: bool = False
    requires_gpu: bool = False
    pinned: bool = True
    warnings: list[str] = field(default_factory=list)
    builder: str | None = None  # name of the command-builder function

    @property
    def out_subdir(self) -> str:
        return f"{self.group}/{self.result_id}"

    @property
    def mode_label(self) -> str:
        """Human-readable mode for menus and --list.

        `mode` stays a bare control value that the code branches on; this is what a reader
        sees. "validate-only" understates what it means for a frozen artifact -- nothing is
        recomputed because recomputing it could not be claimed as identical, which is a
        policy, not a limitation of the run.
        """
        return {"validate-only": "frozen/checksum validation",
                "checkpoint-reanalysis": "frozen checkpoint reanalysis"}.get(self.mode, self.mode)


def _fc(rid, name, family, rep, idx, difficulty, seed_note, run_dir, heavy=False):
    n = f"{idx:02d}"
    latent = rep == "latent8"
    return Result(
        result_id=rid,
        display_name=name,
        group="02_forecasting",
        mode="reproduce",
        difficulty=difficulty,
        status="reproduce",
        action="implemented",
        target_ids=[f"FC-{n}V", f"FC-{n}T", f"FC-{n}S", f"FC-{n}C"],
        package_config_path=f"final_thesis_package_v001/configs/02_forecasting/{rid}.json",
        package_result_path=f"final_thesis_package_v001/results/02_forecasting/{family}/selected/{run_dir}",
        expected_source="repro_validation/expected/validation_targets.csv",
        plots=[
            "validation_relative_l2_horizon.png",
            "validation_spatiotemporal_comparison.png",
            "test_relative_l2_horizon.png",
            "test_spatiotemporal_comparison.png",
        ],
        laptop_safe=not heavy,
        pinned=True,
        warnings=[seed_note],
        builder=f"build_forecasting_{family}",
    )


REGISTRY: list[Result] = [
    Result(
        result_id="simulation_dataset",
        display_name="KSE simulation dataset",
        group="00_simulation",
        mode="validate-only",
        difficulty="fast",
        status="frozen",
        action="implemented_validate_only",
        target_ids=["SIM-001", "SIM-002", "SIM-003", "SIM-004", "SIM-005"],
        package_config_path="final_thesis_package_v001/configs/00_simulation/kse_simulation.json",
        package_result_path="final_thesis_package_v001/results/00_simulation",
        expected_source="repro_validation/expected/simulation_expected.json",
        plots=["kse_spatiotemporal_lyapunov.png"],
        laptop_safe=True,
        warnings=[
            "FROZEN. The solver is never re-run: a chaotic trajectory over ~130 Lyapunov "
            "times cannot be regenerated bit-identically, and every downstream result is "
            "conditioned on these exact bytes. Checksum validation only."
        ],
    ),
    Result(
        result_id="latent8_trial014_representation",
        display_name="AE-SINDy latent8 trial014 representation",
        group="01_ae_sindy",
        mode="validate-only",
        difficulty="fast",
        status="frozen",
        action="implemented_validate_only",
        target_ids=["AE8-001", "AE8-002", "AE8-003", "AE8-004", "AE8-005", "AE8-006"],
        package_config_path="final_thesis_package_v001/configs/01_ae_sindy/latent8_trial014_representation.json",
        package_result_path="final_thesis_package_v001/results/01_ae_sindy/latent8_trial014_representation",
        expected_source="repro_validation/expected/validation_targets.csv",
        plots=[
            "training_history.png",
            "reconstruction.png",
            "xi_heatmap.png",
            "latent_space_comparison.png",
            "rollout_encoder_comparison.png",
            "rollout_sindy_comparison.png",
        ],
        laptop_safe=True,
        requires_gpu=False,
        pinned=False,
        warnings=[
            "FROZEN learned representation. The autoencoder is NOT retrained: exact "
            "cross-hardware GPU retraining is not claimed, so retraining is not offered as "
            "the reproduction path. Checksum/shape validation only, plus the reconstruction "
            "metrics read from the package.",
            "Why frozen rather than GPU-gated: the original was trained on a specific CUDA "
            "device (RTX 2080 Ti, ~hours) WITHOUT the pinned profile. A legitimate retrain "
            "elsewhere need not match bit-for-bit -- the same cross-environment CUDA "
            "nondeterminism that got the latent6 retrain rejected. Offering a retrain as "
            "'reproduction' would promise an identity the hardware cannot deliver.",
            "What this DOES establish: the exact bytes every downstream result was "
            "conditioned on. The posthoc refit and both latent8 forecasting models consume "
            "this frozen representation and reproduce from it identically -- that is the "
            "reproducibility claim, and it is stronger than a retrain would be.",
        ],
        builder=None,  # deliberately none: run_train.py is unreachable from the default official
                       # reproduction path (registry/BUILDERS). It is reachable ONLY through the
                       # explicit opt-in advanced GPU historical-profile provenance mode.
    ),
    Result(
        result_id="latent6_trial002_dynamics_baseline",
        display_name="AE-SINDy latent6 trial002 dynamics baseline",
        group="01_ae_sindy",
        mode="checkpoint-reanalysis",
        difficulty="medium",
        status="frozen",
        action="implemented",
        target_ids=["AE6-001", "AE6-002", "AE6-003", "AE6-004"],
        package_config_path="final_thesis_package_v001/configs/01_ae_sindy/latent6_trial002_dynamics_baseline_checkpoint_reanalysis.json",
        package_result_path="final_thesis_package_v001/results/01_ae_sindy/latent6_trial002_dynamics_baseline",
        expected_source="repro_validation/expected/validation_targets.csv",
        plots=[
            "training_history.png",
            "reconstruction.png",
            "xi_heatmap.png",
            "latent_space_comparison.png",
            "rollout_encoder_comparison.png",
            "rollout_sindy_comparison.png",
        ],
        laptop_safe=True,
        warnings=[
            "FROZEN CHECKPOINT. Never retrained: a from-scratch retrain was attempted and "
            "REJECTED (cross-environment CUDA nondeterminism). run_train.py is not invoked.",
            "Reanalysis reads AND writes its run directory, so the frozen checkpoint is "
            "COPIED into the run folder first and reanalysed there. The package copy is "
            "never touched.",
        ],
        builder="build_ae_sindy_latent6",
    ),
    Result(
        result_id="latent8_trial014_posthoc_predictive",
        display_name="AE-SINDy latent8 post-hoc SINDy refit",
        group="01_ae_sindy",
        mode="reproduce",
        difficulty="medium",
        status="reproduce",
        action="implemented",
        target_ids=["PH-001", "PH-002", "PH-003", "PH-004", "PH-005"],
        package_config_path="final_thesis_package_v001/configs/01_ae_sindy/latent8_trial014_posthoc_thr0025.json",
        package_result_path="final_thesis_package_v001/results/01_ae_sindy/latent8_trial014_posthoc_thr0025_predictive",
        expected_source="repro_validation/expected/validation_targets.csv",
        plots=[],  # the post-hoc refit ships no figures of its own
        laptop_safe=True,
        warnings=[
            "Refits on the FROZEN latent8 representation (read-only). The autoencoder is "
            "not retrained."
        ],
        builder="build_ae_sindy_posthoc",
    ),
    _fc("ngrc_full64", "NGRC full64", "ngrc", "full64", 1, "medium/heavy",
        "Deterministic (no RNG): a seed check is not definable. Its ridge solve is "
        "conditioning-sensitive -- the pinned profile is mandatory, not advisory.",
        "ngrc_full64_selected_v003_stable_ridge"),
    _fc("ngrc_latent8", "NGRC latent8", "ngrc", "latent8", 2, "medium",
        "Deterministic (no RNG): a seed check is not definable.",
        "ngrc_latent8_selected_v002"),
    _fc("rc_full64", "RC full64", "rc", "full64", 3, "medium",
        "Stochastic: locked to the median-representative seed 9403115.",
        "rc_full64_selected_v003_seedrobust"),
    _fc("rc_latent8", "RC latent8", "rc", "latent8", 4, "medium",
        "Stochastic: locked to the median-representative seed 9403102.",
        "rc_latent8_selected_v003_seedrobust"),
    _fc("perc_full64", "PERC full64", "perc", "full64", 5, "medium",
        "Stochastic: locked to the median-representative seed 9403103.",
        "perc_full64_selected_v003_seedrobust"),
    _fc("perc_latent8", "PERC latent8", "perc", "latent8", 6, "medium",
        "Stochastic: locked to the median-representative seed 9403113.",
        "perc_latent8_selected_v002_seedrobust"),
    _fc("pinn_full64", "PINN full64", "pinn", "full64", 7, "heavy",
        "Stochastic: locked to seed 9301114 (candidate C5), fixed before any test access. "
        "Selected with an ACTIVE physics term by a validation-only 10-candidate x 15-seed "
        "rerank, on median validation horizon. Report the median 19.8 with its seed spread "
        "(8.5-23.1), and report the test horizon 7.3 plainly -- it is far below validation "
        "and no retuning followed.",
        "pinn_full64_selected_v003_physics_active", heavy=True),
    _fc("pinn_latent8", "PINN latent8", "pinn", "latent8", 8, "heavy",
        "Stochastic: locked to the median-representative seed 9403108.",
        "pinn_latent8_selected_v002_seedrobust", heavy=True),
]

BY_ID = {r.result_id: r for r in REGISTRY}

# Package-wide targets, checked in --validate-existing rather than owned by one result.
PACKAGE_TARGETS = ["PLOT-001", "TBL-001"]


# ───────────────────────────── command builders ─────────────────────────────
#
# Every command below is transcribed from the package execution report that produced the
# canonical numbers -- never guessed. Sources:
#   forecasting  -> final_thesis_package_v001/manifests/stage4_forecasting_group_ab_sha256_manifest.csv  §2
#                   final_thesis_package_v001/manifests/stage4_forecasting_group_c_pinn_sha256_manifest.csv §1
#   latent8      -> final_thesis_package_v001/manifests/stage3_ae_sindy_latent8_sha256_manifest.csv      §2
#   latent6      -> final_thesis_package_v001/manifests/stage4a_latent6_trial002_checkpoint_reanalysis_sha256_manifest.csv §3
#   post-hoc     -> final_thesis_package_v001/manifests/stage5_ae_sindy_posthoc_sha256_manifest.csv      §3


@dataclass
class Command:
    argv: list[str]
    env: dict[str, str]
    label: str


def _fc_interpreter(result: Result) -> tuple[str, dict[str, str]]:
    """Full-64 non-PINN stays on system python (no torch, no decode); everything that
    decodes latent -> physical, plus both PINNs, needs the conda torch interpreter."""
    latent = result.result_id.endswith("latent8")
    if result.result_id.startswith("pinn") or latent:
        env = {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": f"{REPO_ROOT}/forecasting/src:{REPO_ROOT}/ae_sindy"}
        return CONDA_PY, env
    return SYS_PY, {"PYTHONPATH": f"{REPO_ROOT}/forecasting/src"}


def _build_forecasting(result: Result, cfg: Path, runner: str) -> list[Command]:
    py, env = _fc_interpreter(result)
    return [Command([py, f"forecasting/scripts/{runner}", "--config", str(cfg)], env,
                    f"{result.display_name} selected evaluation")]


def build_forecasting_ngrc(r, cfg): return _build_forecasting(r, cfg, "run_ngrc.py")
def build_forecasting_rc(r, cfg): return _build_forecasting(r, cfg, "run_rc.py")
def build_forecasting_perc(r, cfg): return _build_forecasting(r, cfg, "run_perc.py")
def build_forecasting_pinn(r, cfg): return _build_forecasting(r, cfg, "run_pinn.py")


def _RETIRED_build_ae_sindy_latent8(result: Result, cfg: Path) -> list[Command]:
    """*** NOT A DEFAULT PATH. NOT DISPATCHABLE FROM THE REGISTRY. ***

    The GPU retrain recipe that originally produced the latent8 representation, kept as a
    record of how the canonical numbers were made -- not as a DEFAULT reproduction option.

    latent8_trial014_representation is mode="validate-only" with builder=None, so this
    function is unreachable from the DEFAULT path: it is absent from BUILDERS, so naming it
    in a registry entry raises KeyError rather than silently training. That is deliberate.

    The retrain IS available, but ONLY as the explicit opt-in advanced provenance mode
    (--advanced-gpu-latent8-historical, see advanced_execute), which builds its own commands
    directly, writes only under test_repro_ae_sindy/, and is never selected by --all-*. This
    retired builder itself stays unused even by that path.

    Why it must not be the default: the original ran on a specific CUDA device without the
    pinned profile. A retrain elsewhere need not reproduce bit-for-bit (cross-environment
    CUDA nondeterminism -- the same failure that got the latent6 retrain rejected), so
    offering it as DEFAULT "reproduction" would promise an identity the hardware cannot
    deliver. The advanced path frames it as same-stack provenance, never a portable claim.
    """
    env = {"PYTHONPATH": f"{REPO_ROOT}/ae_sindy"}
    return [
        Command([CONDA_PY, "ae_sindy/scripts/run_train.py", "--config", str(cfg)], env, "train"),
        Command([CONDA_PY, "ae_sindy/scripts/run_analysis.py", "--config", str(cfg)], env, "analysis"),
        Command([CONDA_PY, "ae_sindy/scripts/run_visualize.py", "--config", str(cfg)], env, "visualize"),
    ]


def build_ae_sindy_latent6(result: Result, cfg: Path) -> list[Command]:
    env = {"PYTHONPATH": f"{REPO_ROOT}/ae_sindy", "CUDA_VISIBLE_DEVICES": ""}
    return [
        Command([CONDA_PY, "ae_sindy/scripts/run_analysis.py", "--config", str(cfg),
                 "--eval-split", "test"], env, "checkpoint reanalysis"),
        Command([CONDA_PY, "ae_sindy/scripts/run_visualize.py", "--config", str(cfg)], env, "visualize"),
    ]


def build_ae_sindy_posthoc(result: Result, cfg: Path) -> list[Command]:
    env = {"PYTHONPATH": f"{REPO_ROOT}/ae_sindy", "CUDA_VISIBLE_DEVICES": ""}
    return [
        Command([CONDA_PY, "scripts/posthoc_refit_sindy.py", "--config", str(cfg),
                 "--windows-preset", "test"], env, "post-hoc SINDy refit"),
    ]


# _RETIRED_build_ae_sindy_latent8 is deliberately absent: latent8_trial014_representation is
# a frozen/checksum-validated representation, never retrained on the DEFAULT path. Its omission
# here is what makes run_train.py unreachable from the default reproduction -- a registry entry
# naming it would KeyError, loudly, instead of quietly starting a GPU training run that could not
# be claimed as identical anyway. run_train.py is reachable ONLY via the opt-in advanced provenance
# mode (advanced_execute), which is same-stack-only and writes solely under test_repro_ae_sindy/.
BUILDERS: dict[str, Callable[[Result, Path], list[Command]]] = {
    "build_forecasting_ngrc": build_forecasting_ngrc,
    "build_forecasting_rc": build_forecasting_rc,
    "build_forecasting_perc": build_forecasting_perc,
    "build_forecasting_pinn": build_forecasting_pinn,
    "build_ae_sindy_latent6": build_ae_sindy_latent6,
    "build_ae_sindy_posthoc": build_ae_sindy_posthoc,
}


def child_env(result: Result, extra: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    if result.pinned:
        env.update(PINNED_ENV)
    else:
        env["MPLBACKEND"] = "Agg"
    env.update(extra)
    return env


# ───────────────────────────── config redirection ─────────────────────────────


def redirect_config(result: Result, run_dir: Path) -> tuple[Path, list[str]]:
    """Copy the package config into the run folder with its OUTPUT paths redirected.

    The package configs write back into final_thesis_package_v001/. Handing one to a
    runner unchanged would mutate the frozen package, so redirection is mandatory, not a
    convenience. Input/data paths are left pointing at the package: those are reads.
    """
    src = REPO_ROOT / result.package_config_path
    cfg = json.loads(src.read_text())
    notes: list[str] = []

    dest_results = run_dir / "results" / result.group / result.result_id
    dest_results.mkdir(parents=True, exist_ok=True)

    if result.group == "02_forecasting":
        cfg.setdefault("output", {})
        notes.append(f"output.base_dir: {cfg['output'].get('base_dir')} -> {dest_results}")
        cfg["output"]["base_dir"] = str(dest_results)
        cfg["output"]["append_registry"] = False  # never touch the registry
        notes.append("output.append_registry pinned to false")
    elif result.result_id == "latent8_trial014_posthoc_predictive":
        notes.append(f"output_root: {cfg.get('output_root')} -> {dest_results}")
        cfg["output_root"] = str(dest_results)
    else:  # AE-SINDy train / reanalysis configs
        notes.append(f"output_dir: {cfg.get('output_dir')} -> {dest_results}")
        cfg["output_dir"] = str(dest_results)
        if cfg.get("visualization_output_dir"):
            cfg["visualization_output_dir"] = str(dest_results)

    dest_cfg = run_dir / "configs" / result.group / Path(result.package_config_path).name
    dest_cfg.parent.mkdir(parents=True, exist_ok=True)
    dest_cfg.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    return dest_cfg, notes


def stage_frozen_checkpoint(result: Result, run_dir: Path) -> list[str]:
    """latent6 reanalysis reads AND writes its run dir, so the frozen checkpoint must be
    copied into the run folder before reanalysis. Copy-out, never write-in."""
    src = REPO_ROOT / result.package_result_path / "latent6_trial002_dynamics_baseline"
    dst = run_dir / "results" / result.group / result.result_id / "latent6_trial002_dynamics_baseline"
    assert_outside_package(dst, "the staged checkpoint")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    # The package run dir already contains analysis_summary*.pkl and metrics.pkl. If
    # the reanalysis ever exited 0 without rewriting them, resolve_observed()'s rglob
    # would find exactly one hit -- the copied PACKAGE file -- and the checks would
    # pass without the child having produced anything. Removing them from the staged
    # copy makes that failure mode impossible rather than merely unlikely.
    stale = sorted(dst.rglob("analysis_summary*.pkl")) + sorted(dst.rglob("metrics.pkl"))
    for f in stale:
        f.unlink()
    notes = [f"copied frozen checkpoint {src.relative_to(REPO_ROOT)} -> run folder (read-only source)"]
    if stale:
        notes.append(
            "removed {} pre-existing summary file(s) from the staged copy so the "
            "reanalysis must regenerate them: {}".format(
                len(stale), ", ".join(f.name for f in stale)))
    return notes


# ───────────────────────────── contract loading ─────────────────────────────


def load_policy() -> dict:
    return json.loads(POLICY_JSON.read_text())


def load_targets() -> dict[str, dict]:
    with TARGETS_CSV.open() as fh:
        return {r["target_id"]: r for r in csv.DictReader(fh)}


def parse_locator(spec: str) -> list[tuple[Path, str | None]]:
    """Parse the Stage 9 `path :: field` locator form.

    Entries are separated by ' | '. A bare filename in a later entry is a sibling of the
    first entry's directory (e.g. 'z_train.npy :: shape | z_valid.npy :: shape').
    """
    out: list[tuple[Path, str | None]] = []
    base: Path | None = None
    for chunk in spec.split(" | "):
        chunk = chunk.strip()
        if not chunk:
            continue
        if " :: " in chunk:
            raw, fld = chunk.split(" :: ", 1)
            raw, fld = raw.strip(), fld.strip()
        else:
            raw, fld = chunk, None
        p = Path(raw)
        if not p.is_absolute():
            p = (REPO_ROOT / p) if "/" in raw else ((base.parent / raw) if base else REPO_ROOT / raw)
        if base is None:
            base = p
        out.append((p, fld))
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dotted(obj: Any, key: str) -> Any:
    for part in key.split("."):
        if not isinstance(obj, dict) or part not in obj:
            raise KeyError(key)
        obj = obj[part]
    return obj


def read_field(path: Path, fld: str | None) -> Any:
    """Read a value out of a package artifact. Handles the four container types the
    contract points at: .json, .csv, .pkl and .npy."""
    suf = path.suffix.lower()
    if suf == ".json":
        return dotted(json.loads(path.read_text()), fld)
    if suf == ".csv":
        with path.open() as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise ValueError(f"empty csv: {path}")
        if fld not in rows[0]:
            raise KeyError(f"{fld} not in {path.name}: {list(rows[0])}")
        vals = [r[fld] for r in rows]
        return vals[0] if len(vals) == 1 else vals
    if suf == ".pkl":
        with path.open("rb") as fh:
            obj = pickle.load(fh)
        return dotted(obj, fld) if isinstance(obj, dict) else obj
    if suf == ".npy":
        import numpy as np

        return tuple(np.load(path, mmap_mode="r").shape)
    raise ValueError(f"cannot read {path}")


def parse_shape(s: str) -> list[tuple[int, ...]]:
    out = []
    for part in s.split("/"):
        part = part.strip().strip("()")
        if part:
            out.append(tuple(int(x) for x in part.split(",") if x.strip()))
    return out


def as_float(v: Any) -> float:
    if isinstance(v, (list, tuple)):
        raise TypeError("expected scalar")
    return float(v)


# ───────────────────────────── fresh runtime seed ─────────────────────────────


def _rel(p: Path) -> str:
    """Repo-relative path for display, falling back to absolute rather than raising."""
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


# Where each stochastic family's runner records the seed it ACTUALLY consumed, inside its
# own fresh summary.json. Every entry here was read off a real reproduced run, not guessed.
#
# These are RUNTIME records, not config echoes -- and that distinction is the entire point.
# pinn also carries metadata.final_seed_choice.random_seed, which is deliberately NOT used:
# metadata is copied verbatim out of the config, so reading it would re-introduce exactly
# the assumption this gate exists to remove. Only a field the runner wrote from the seed it
# fed to the RNG can witness what the run actually did.
RUNTIME_SEED_FIELDS: dict[str, str] = {
    "rc": "rc_parameters.random_seed",
    "perc": "perc_parameters.random_seed",
    "pinn": "pinn.training.random_seed",
}


def extract_fresh_runtime_seed(result: Result, observed_root: Path) -> tuple[int | None, str]:
    """Read the seed the runner actually used out of this run's fresh summary.

    Returns (seed, diagnostic). A None seed means it could not be established -- an unknown
    seed is NOT a satisfied precondition, and the caller must refuse the 0.11 band.
    """
    family = result.result_id.split("_")[0]
    field = RUNTIME_SEED_FIELDS.get(family)
    if field is None:
        return None, (f"no known runtime-seed schema for family '{family}'; cannot confirm "
                      f"the locked seed from fresh output")
    base = observed_root / "results" / result.group / result.result_id
    hits = sorted(base.rglob("summary.json"))
    if len(hits) != 1:
        return None, (f"expected exactly one fresh summary.json under "
                      f"{result.out_subdir}/, found {len(hits)}")
    # Read and display-format separately: folding both into one try would let a path
    # formatting slip masquerade as "the seed is unreadable", and a wrong seed must be
    # reported AS a wrong seed.
    try:
        raw = read_field(hits[0], field)
    except Exception as exc:  # noqa: BLE001
        return None, f"could not read {field} from fresh summary: {exc!r}"
    try:
        seed = int(raw)
    except (TypeError, ValueError):
        return None, f"{field} in fresh summary is not an integer: {raw!r}"
    return seed, f"{_rel(hits[0])} :: {field}"


def verify_locked_seed(result: Result, targets: dict, observed_root: Path | None
                       ) -> tuple[bool | None, str]:
    """Mechanically confirm the fresh run consumed the contract's locked seed.

    The 0.11 runtime band is only defensible at the locked seed -- pinn_full64 spans
    8.5-23.1 across its 15 seeds (~130x), which dwarfs the band by three orders of
    magnitude. So this precondition must be WITNESSED, never assumed: reading the seed back
    out of the config the wrapper itself wrote would only prove the wrapper can copy a file.

    Returns (verified, note):
      None  -> not applicable; the contract declares this model deterministic (no RNG)
      True  -> fresh output independently records exactly the expected seed
      False -> seed missing, unparseable, or different; the 0.11 band must be refused
    """
    seed_targets = [t for t in result.target_ids
                    if t in targets and targets[t].get("expected_units") == "seed"]
    if not seed_targets:
        # Stochastic-by-default: no seed target means nothing pins the seed, so there is
        # nothing to witness. Refuse rather than pass silently.
        return False, "contract defines no seed target for this result; runtime seed unverifiable"

    tgt = targets[seed_targets[0]]
    if tgt["comparison_method"] == "not_applicable":
        return None, "deterministic model (no RNG): seed precondition not applicable"
    if observed_root is None:
        return False, "no fresh output available to read the runtime seed from"
    try:
        expected = int(tgt["expected_value"])
    except (TypeError, ValueError):
        return False, f"contract seed {tgt['expected_value']!r} is not an integer"

    got, diag = extract_fresh_runtime_seed(result, observed_root)
    if got is None:
        return False, diag
    if got != expected:
        return False, (f"fresh output ran seed {got}, but the contract locks {expected} "
                       f"({diag})")
    return True, f"fresh output independently records seed {got} ({diag})"


# ───────────────────────────── tolerance resolution ─────────────────────────────


def tolerance_band(level: dict, want: float, got: float) -> float:
    """The permitted absolute delta for one comparison.

    Absolute-only levels return their fixed band unchanged -- horizons stay on their dt=0.1
    reasoning and are untouched by this. A level that also carries `tolerance_relative`
    scales its band with the magnitude being compared: float32-derived recomputes carry only
    ~7 significant digits, so a fixed 1e-12 would test the storage format rather than the
    science. The absolute term then acts purely as the floor for the degenerate expected~=0
    case, where a relative band would collapse to nothing.
    """
    abs_tol = level.get("tolerance") or 0.0
    rel = level.get("tolerance_relative")
    if rel is None:
        return abs_tol
    return max(abs_tol, rel * max(abs(want), abs(got)))


def resolve_tolerance(target: dict, policy: dict, reran: bool, pinned: bool,
                      locked_seed: bool | None) -> tuple[str, dict, str | None]:
    """Pick the effective tolerance level and decide whether it may be applied.

    Returns (level_name, level_dict, blocker). A non-None blocker means the comparison
    must be reported REVIEW_REQUIRED rather than PASS/FAIL.

    The two gates below are the whole point of Stage 9's review_required flags:

      horizon_reporting_physical (0.05) is a REPORTING tolerance -- it compares two
      values derived from the same locked computation. For a FRESH run the contract
      requires horizon_runtime_physical (0.11) instead, and that one is only defensible
      under a pinned profile AND the locked seed. Unpinned, ngrc_full64 alone drifts
      ~1.10 (10x); at a different seed pinn_full64 spans 8.5-23.1 (~130x).

      locked_seed is tri-state, from verify_locked_seed(): True (witnessed in fresh
      output), None (contract says the model has no RNG, so the precondition does not
      apply), or False (missing/unparseable/different -- refuse the band).

      strict_float (1e-12) is valid only for read-from-file comparison. The latent
      arrays are float32 (~7 significant digits), so a RECOMPUTED metric can never
      reach 1e-12.
    """
    key = target["tolerance_key"]
    levels = policy["levels"]

    if reran and key == "horizon_reporting_physical":
        key = "horizon_runtime_physical"  # fresh run: the contract forbids the reporting band

    level = levels.get(key, {})
    blocker = None

    if level.get("review_required"):
        if key == "horizon_runtime_physical":
            if not reran:
                blocker = ("horizon_runtime_physical applies only to a controlled rerun; "
                           "this value was read from a file")
            elif not pinned:
                blocker = ("tolerance 0.11 requires the pinned deterministic profile "
                           "(OMP_NUM_THREADS=1); ngrc_full64 drifts ~1.10 unpinned -- 10x")
            elif locked_seed is False:  # None == deterministic model, precondition n/a
                blocker = ("tolerance 0.11 requires the exact locked seed, witnessed in "
                           "fresh output; pinn_full64 spans 8.5-23.1 across seeds -- ~130x")
        elif key == "strict_float":
            if reran:
                blocker = ("strict_float 1e-12 is valid only for read-from-file values; "
                           "the latent arrays are float32, so a recomputed metric cannot "
                           "reach 1e-12. recomputed_float_relative (1e-6 relative) covers "
                           "pinned CPU recomputes from frozen inputs, but deliberately NOT "
                           "this target: latent8_trial014_representation is GPU-gated and "
                           "unpinned, and a legitimate retrain need not match to 1e-6 "
                           "either -- this needs a judgement, not a wider band")
    return key, level, blocker


# ───────────────────────────── validation ─────────────────────────────


@dataclass
class Check:
    target_id: str
    result_id: str
    what: str
    expected: str
    observed: str
    status: str
    tolerance_key: str
    note: str = ""
    observed_source: str = ""  # the file actually read -- fresh output, or the package

    def row(self) -> dict:
        return {
            "target_id": self.target_id, "result_id": self.result_id, "what": self.what,
            "expected": self.expected, "observed": self.observed, "status": self.status,
            "tolerance_key": self.tolerance_key, "observed_source": self.observed_source,
            "note": self.note,
        }


def _declared_count(expected_value: str) -> int | None:
    """The file count a contract target declares, e.g. '76 PNGs, 76/76 sha256 match'."""
    m = re.match(r"\s*(\d+)\s", expected_value or "")
    return int(m.group(1)) if m else None


def _package_check(tid: str, target: dict) -> Check:
    """PLOT-001 / TBL-001: manifest-driven checksum sweeps."""
    mk = lambda status, obs, note="": Check(tid, "package", target["artifact_type"],
                                            target["expected_value"], obs, status,
                                            target["tolerance_key"], note)
    manifests: list[Path] = []
    if tid == "PLOT-001":
        manifests = [p for p, _ in parse_locator(target["expected_file_or_metric"])]
        col_file, col_hash = "file", "sha256"
    else:
        manifests = [TABLE_MANIFEST]
        col_file, col_hash = "relative_path", "sha256"

    total = bad = 0
    listed: set[str] = set()
    for m in manifests:
        if not m.exists():
            return mk(ERROR, f"missing manifest {m}")
        with m.open() as fh:
            for row in csv.DictReader(fh):
                total += 1
                listed.add(row[col_file])
                p = REPO_ROOT / row[col_file]
                if not p.exists() or sha256(p) != row[col_hash]:
                    bad += 1
    obs = f"{total - bad}/{total} sha256 match"

    # A manifest sweep alone cannot see two failure modes: a manifest that has been
    # truncated (it would report a clean 10/10), and a figure shipped in the package
    # but listed nowhere. Both are checked explicitly.
    declared = _declared_count(target["expected_value"])
    if declared is not None and declared != total:
        return mk(FAIL, f"{obs}; manifest lists {total} files, contract declares {declared}")

    if tid == "PLOT-001":
        on_disk = {p.relative_to(REPO_ROOT).as_posix()
                   for p in (PACKAGE / "plots").rglob("*.png")}
        unlisted = sorted(on_disk - listed)
        if unlisted:
            return mk(FAIL,
                      f"{obs}; {len(unlisted)} figure(s) present but in no manifest: "
                      + ", ".join(unlisted[:3]) + ("..." if len(unlisted) > 3 else ""))

    return mk(PASS if bad == 0 else FAIL, obs)


def resolve_observed(locs: list[tuple[Path, str | None]], result: Result | None,
                     observed_root: Path | None, reran: bool
                     ) -> tuple[list[tuple[Path, str | None]] | None, str | None]:
    """Point a reproduced result's checks at ITS OWN fresh output.

    A reproduced result must be validated against what this run actually produced. Reading
    the packaged artifact here would compare the package against itself and PASS
    trivially -- a green result that tested nothing.

    Only artifacts under the package RESULTS tree are remapped. Config checksums and seed
    fields deliberately keep pointing at the package config: that is the run's INPUT, and
    confirming it is unmodified is exactly what those targets are for.

    The fresh tree is nested differently from the package tree (the runner picks its own
    run_id), so files are located by name under the result's fresh output directory. An
    ambiguous or missing match is an ERROR, never a silent fallback to the package.
    """
    if not (reran and observed_root and result):
        return locs, None
    pkg_results = PACKAGE / "results"
    base = observed_root / "results" / result.group / result.result_id
    out: list[tuple[Path, str | None]] = []
    for p, f in locs:
        try:
            p.relative_to(pkg_results)
        except ValueError:
            out.append((p, f))  # not a result artifact (e.g. a config) -- leave it alone
            continue
        hits = sorted(base.rglob(p.name))
        if len(hits) != 1:
            return None, (f"expected exactly one fresh {p.name} under "
                          f"{base.relative_to(REPO_ROOT)}/, found {len(hits)}")
        out.append((hits[0], f))
    return out, None


def validate_target(tid: str, target: dict, policy: dict, result: Result | None,
                    reran: bool, observed_root: Path | None,
                    locked_seed: bool | None = None,
                    locked_seed_note: str = "") -> Check:
    """Validate one contract target. Never raises: an unreadable target is ERROR.

    locked_seed/locked_seed_note come from verify_locked_seed() and are computed once per
    result by the caller; they only affect the horizon_runtime_physical gate.
    """
    if tid in PACKAGE_TARGETS:
        return _package_check(tid, target)

    rid = result.result_id if result else target["result_id"]
    exp_raw = target["expected_value"]
    method = target["comparison_method"]
    what = (target["expected_file_or_metric"].split(" :: ")[-1]
            if " :: " in target["expected_file_or_metric"]
            else Path(target["expected_file_or_metric"]).name)

    src_seen: list[str] = []  # filled once the locator is resolved, for the audit trail
    mk = lambda status, obs, tk, note="": Check(tid, rid, what, exp_raw, str(obs), status, tk,
                                                note, " | ".join(src_seen))

    pinned = bool(result and result.pinned)
    tkey, level, blocker = resolve_tolerance(target, policy, reran, pinned, locked_seed)

    if method == "not_applicable":
        # For the deterministic NGRC models the contract expects NO seed field. That is a
        # real assertion, not a no-op: a seed appearing here would mean the model gained
        # an RNG and the "deterministic" claim would need re-examining.
        try:
            path, fld = parse_locator(target["expected_file_or_metric"])[0]
            present = True
            try:
                read_field(path, fld)
            except (KeyError, ValueError):
                present = False
            if present:
                return mk(FAIL, f"{fld} unexpectedly present", tkey,
                          "contract says this model is deterministic (no RNG)")
            return mk(SKIPPED, "field absent, as expected", tkey,
                      "deterministic model: a seed check is not definable")
        except Exception as exc:  # noqa: BLE001
            return mk(ERROR, repr(exc), tkey)

    try:
        locs = parse_locator(target["expected_file_or_metric"])
        locs, remap_err = resolve_observed(locs, result, observed_root, reran)
        if remap_err:
            return mk(ERROR, remap_err, tkey,
                      "a reproduced result is validated against its own fresh output, "
                      "never against the package artifact it reproduces")
        src_seen.extend(str(p.relative_to(REPO_ROOT)) for p, _ in locs)
        for path, _ in locs:
            if not path.exists():
                return mk(ERROR, f"missing file: {path}", tkey)

        if method == "sha256_exact":
            obs = sha256(locs[0][0])
            return mk(PASS if obs == exp_raw else FAIL, obs, tkey)

        if method == "shape_exact":
            shapes = [read_field(p, f) for p, f in locs]
            want = parse_shape(exp_raw)
            ok = len(shapes) == len(want) and all(a == b for a, b in zip(shapes, want))
            return mk(PASS if ok else FAIL, " / ".join(str(s) for s in shapes), tkey)

        if method == "integer_exact":
            if target["artifact_type"] == "split_indices":
                # Derive the split sizes rather than restating them: the solver config
                # gives the post-transient step count, and the contract gives the split
                # boundaries. This catches both a changed simulation length and splits
                # that no longer partition the series exactly.
                cfg = json.loads(locs[0][0].read_text())
                n = cfg["n_steps"] - cfg["transient"]
                spans = json.loads((CONTRACT / "expected" / "simulation_expected.json").read_text())
                spans = spans["split_indices"]
                bounds = [spans["train"], spans["validation"], spans["test"]]
                obs_v = [b - a for a, b in bounds]
                want = [int(x) for x in exp_raw.split("/")]
                ok = (obs_v == want and bounds[0][0] == 0 and bounds[-1][1] == n
                      and all(bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1)))
                note = "" if ok else (f"config post-transient steps = {n}; splits "
                                      f"{bounds} must partition [0, {n}) contiguously")
                return mk(PASS if ok else FAIL, " / ".join(map(str, obs_v)), tkey, note)
            raw = read_field(*locs[0])
            obs_v = int(float(raw if not isinstance(raw, list) else raw[0]))
            return mk(PASS if obs_v == int(exp_raw) else FAIL, obs_v, tkey)

        if method == "numeric_abs_tolerance":
            raw = read_field(*locs[0])
            if blocker:
                obs = raw if not isinstance(raw, list) else ", ".join(map(str, raw))
                return mk(REVIEW, obs, tkey, blocker)
            tol = level.get("tolerance")
            if tol is None:
                return mk(ERROR, "no tolerance in policy", tkey)
            if "|" in exp_raw:  # multi-value target (post-hoc per-window horizons)
                want = [float(x) for x in exp_raw.split("|")]
                got = [float(x) for x in (raw if isinstance(raw, list) else [raw])]
                ok = len(got) == len(want) and all(
                    abs(a - b) <= tolerance_band(level, b, a) for a, b in zip(got, want))
                return mk(PASS if ok else FAIL, " | ".join(f"{g:g}" for g in got), tkey)
            got = as_float(raw)
            want_v = float(exp_raw)
            ok = abs(got - want_v) <= tolerance_band(level, want_v, got)
            # Record HOW the 0.11 band was earned, not just that it was: the seed evidence
            # is the audit trail for the gate's strongest precondition.
            note = locked_seed_note if tkey == "horizon_runtime_physical" else ""
            return mk(PASS if ok else FAIL, f"{got:.12g}", tkey, note)

        return mk(ERROR, f"unknown comparison_method {method}", tkey)
    except Exception as exc:  # noqa: BLE001
        return mk(ERROR, repr(exc), tkey)


# ───────────────────────────── run folder ─────────────────────────────


def environment_snapshot(pinned_applied: bool, notes: list[str]) -> str:
    lines = [
        "Environment snapshot",
        "====================",
        f"timestamp        : {_dt.datetime.now().isoformat(timespec='seconds')}",
        f"repo root        : {REPO_ROOT}",
        f"launch python    : {sys.version.split()[0]} ({sys.executable})",
        f"platform         : {sys.platform}",
        "",
        "Deterministic profile (H-03, scripts/repro_env.sh) applied to child processes:",
    ]
    for k, v in PINNED_ENV.items():
        inherited = os.environ.get(k)
        mark = "" if inherited == v else f"   (parent shell had: {inherited or 'unset'})"
        lines.append(f"  {k}={v}{mark}")
    lines += [
        "",
        f"pinned profile applied to this run : {pinned_applied}",
        "",
        "Interpreters:",
        f"  system : {SYS_PY}  (full-64 forecasting; numpy/scipy, no decode)",
        f"  conda  : {CONDA_PY}  (torch 2.6.0; latent decode, PINN, AE-SINDy)",
        "",
        "Notes:",
    ]
    lines += [f"  - {n}" for n in notes] or ["  (none)"]
    return "\n".join(lines) + "\n"


def write_validation(run_dir: Path, checks: list[Check]) -> None:
    vdir = run_dir / "validation"
    vdir.mkdir(parents=True, exist_ok=True)
    cols = ["target_id", "result_id", "what", "expected", "observed", "status",
            "tolerance_key", "observed_source", "note"]
    with (vdir / "validation_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for c in checks:
            w.writerow(c.row())
    summary = {s: sum(1 for c in checks if c.status == s) for s in (PASS, FAIL, REVIEW, SKIPPED, ERROR)}
    (vdir / "validation_results.json").write_text(
        json.dumps({"summary": summary, "checks": [c.row() for c in checks]}, indent=2) + "\n")


@functools.lru_cache(maxsize=1)
def _plot_manifest_index() -> dict[str, str]:
    """repo-relative plot path -> recorded sha256, from the stage6/stage7b manifests."""
    index: dict[str, str] = {}
    for name in ("stage6_final_plots_sha256_manifest.csv",
                 "stage7b_seed_robust_plots_sha256_manifest.csv"):
        man = PACKAGE / "manifests" / name
        if not man.exists():
            continue
        with man.open() as fh:
            for row in csv.DictReader(fh):
                index[row["file"]] = row["sha256"]
    return index


def plot_manifest_digest(path: Path) -> str | None:
    """The digest a manifest records for `path`, or None if it lists no such file."""
    try:
        rel = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return None
    return _plot_manifest_index().get(rel)


def copy_plots(result: Result, run_dir: Path) -> list[tuple[str, str, str]]:
    """Copy the main package plots for a result into the run folder.

    Every plot copied here is a checksum-verified copy of the package figure, labelled
    package_verified_copy. It shows the packaged run, NOT the fresh output -- claiming
    otherwise would be false.

    This function is the only path that populates plots/. Separately, some runners write
    figures into their own results/ dir as a side effect (latent6's run_visualize.py does;
    its six regenerated figures happened to come out byte-identical to the packaged ones).
    Those are NOT package plot reproduction targets and are not claimed as such unless a
    contract target explicitly validates them.
    """
    if not result.plots:
        return []
    if result.group == "00_simulation":
        src_dir = PACKAGE / "plots" / "00_simulation"
    elif result.group == "01_ae_sindy":
        src_dir = PACKAGE / "plots" / "01_ae_sindy" / result.result_id
    else:
        src_dir = PACKAGE / "plots" / "02_forecasting" / "per_model" / result.result_id

    dst_dir = run_dir / "plots" / result.group / result.result_id
    out = []
    for name in result.plots:
        src = src_dir / name
        if not src.exists():
            out.append((name, "MISSING", "not present in package"))
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / name)
        digest = sha256(src)
        recorded = plot_manifest_digest(src)
        if recorded is None:
            label = "UNLISTED_IN_MANIFEST"
        elif recorded != digest:
            label = "MANIFEST_MISMATCH"
        else:
            label = "package_verified_copy"
        out.append((str(dst_dir.relative_to(run_dir) / name), label, digest[:12]))
    return out


def run_commands(result: Result, cmds: list[Command], run_dir: Path) -> tuple[bool, list[str]]:
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    ran: list[str] = []
    for i, cmd in enumerate(cmds, 1):
        log = logs / f"{result.result_id}_{i:02d}_{cmd.label.split()[0]}.log"
        logger.info("Running %s", cmd.label)
        logger.debug("command: %s", " ".join(cmd.argv))
        logger.debug("child log -> %s", log.relative_to(REPO_ROOT))

        # The child inherits this terminal, so its structured logs appear live on
        # stderr beneath the workflow heading, and it is handed its own
        # deterministic per-component --log-file so the full record is still
        # persisted. No two processes ever write the same file, and no pipe sits
        # between parent and child (nothing to dead-lock or duplicate).
        log.write_text("$ " + " ".join(cmd.argv) + "\n\n")
        child_argv = [*cmd.argv, "--log-file", str(log)]
        sys.stdout.flush()
        sys.stderr.flush()
        proc = subprocess.run(child_argv, cwd=REPO_ROOT, env=child_env(result, cmd.env))
        ran.append(" ".join(cmd.argv))
        logger.debug("%s exit code %d", cmd.label, proc.returncode)
        if proc.returncode != 0:
            logger.error("%s FAILED (exit %d); see %s",
                         cmd.label, proc.returncode, log.relative_to(REPO_ROOT))
            return False, ran
        logger.info("%s completed", cmd.label)
    return True, ran


def reproduced_summary(result: Result, run_dir: Path) -> Path | None:
    base = run_dir / "results" / result.group / result.result_id
    hits = sorted(base.rglob("summary.json"))
    return hits[0] if hits else None


# ───────────────────────────── report ─────────────────────────────


def _md_cell(value: str) -> str:
    """Escape a value for display inside a markdown table cell.

    Multi-value contract targets separate their values with '|' (e.g. PH-002's five
    per-window horizons), which collides with the markdown column delimiter and splits
    one cell into several. Display-only: the CSV/JSON in write_validation() are built
    from Check.row() and never pass through here.
    """
    return value.replace("|", r"\|")


def write_report(run_dir: Path, request: dict, results: list[Result], checks: list[Check],
                 per_result: dict, env_notes: list[str], validate_existing: bool) -> None:
    n = {s: sum(1 for c in checks if c.status == s) for s in (PASS, FAIL, REVIEW, SKIPPED, ERROR)}
    L: list[str] = []
    A = L.append
    A("# Reproduction report")
    A("")
    A(f"**Run:** `{run_dir.name}` · **Created:** {request['created']}")
    A(f"**Mode:** {'validate existing package (no models run)' if validate_existing else 'selected-result reproduction'}")
    A("")
    A(f"**Validation:** {n[PASS]} PASS · {n[FAIL]} FAIL · {n[REVIEW]} REVIEW_REQUIRED · "
      f"{n[SKIPPED]} SKIPPED · {n[ERROR]} ERROR")
    A("")
    A("Expected values and tolerances come from the Stage 9 contract under `repro_validation/`; ")
    A("this run did not define any of its own.")
    A("")
    A("## Requested results")
    A("")
    if validate_existing:
        A(f"Package-wide validation: every contract target for all {len(REGISTRY)} final results was ")
        A("checked against the packaged artifacts. No individual result was selected, and no model ran.")
    else:
        A("| result | mode | difficulty | status |")
        A("|---|---|---|---|")
        for r in results:
            A(f"| `{r.result_id}` | {r.mode} | {r.difficulty} | "
              f"{per_result.get(r.result_id, {}).get('status', 'n/a')} |")
    A("")
    A("## Environment")
    A("")
    A("See [`environment_snapshot.txt`](environment_snapshot.txt). The H-03 deterministic profile ")
    A("(`OMP_NUM_THREADS=1` and friends, `PYTHONHASHSEED=0`) is applied to child processes; it is ")
    A("load-bearing rather than cosmetic, because `ngrc_full64`'s horizon moves ~0.29% (≈1.10 at 379.9) ")
    A("between pinned and unpinned runs.")
    for note in env_notes:
        A(f"- {note}")
    A("")
    if not validate_existing:
        A("## Commands run")
        A("")
        any_cmd = False
        for r in results:
            for c in per_result.get(r.result_id, {}).get("commands", []):
                A(f"- `{c}`")
                any_cmd = True
        if not any_cmd:
            A("_None — no result in this run executed a model._")
        A("")
        A("## Configs copied")
        A("")
        A("Package configs write their outputs back into `final_thesis_package_v001/`, so each config ")
        A("was copied into this run folder with its output path redirected. The package is never written to.")
        A("")
        for r in results:
            for note in per_result.get(r.result_id, {}).get("config_notes", []):
                A(f"- `{r.result_id}`: {note}")
        A("")
    A("## Plots included")
    A("")
    plot_rows = [row for r in results for row in per_result.get(r.result_id, {}).get("plots", [])]
    if not plot_rows:
        A("_None._ " + ("Validate-existing checks the packaged figures by checksum in place "
                        "(target `PLOT-001`) rather than copying them; the package already holds them."
                        if validate_existing else
                        "No requested result ships figures of its own."))
    else:
        A("| plot | source | sha256 |")
        A("|---|---|---|")
        for plot_name, plot_src, plot_hash in plot_rows:
            A(f"| `{plot_name}` | {plot_src} | `{plot_hash}` |")
        A("")
        A("**`package_verified_copy` means exactly that:** the figure was copied from the package and ")
        A("its sha256 was checked against the packaged plot manifest before the label was applied. It ")
        A("depicts the packaged run, **not** the fresh output of this run. A copy whose digest does not ")
        A("match its manifest record is labelled `MANIFEST_MISMATCH`, and one the manifests do not list ")
        A("at all is labelled `UNLISTED_IN_MANIFEST`; neither is a verified copy. ")
        A("Some runners may also create figures inside `results/` as side effects; those side-effect ")
        A("figures are not counted as package plot reproduction targets unless explicitly validated.")
    A("")
    A("## Validation")
    A("")
    A("| target | result | what | expected | observed | status |")
    A("|---|---|---|---|---|---|")
    for c in checks:
        exp = c.expected if len(c.expected) <= 46 else c.expected[:43] + "…"
        obs = c.observed if len(c.observed) <= 46 else c.observed[:43] + "…"
        A(f"| `{c.target_id}` | `{c.result_id}` | {_md_cell(c.what)} | {_md_cell(exp)} | "
          f"{_md_cell(obs)} | **{c.status}** |")
    A("")
    if not validate_existing:
        A("Each check records the file it actually read in the `observed_source` column of ")
        A("[`validation/validation_results.csv`](validation/validation_results.csv). A reproduced ")
        A("result's metrics are read from **this run's fresh output**, never from the packaged ")
        A("artifact it reproduces — comparing the package against itself would pass trivially and ")
        A("prove nothing. Config checksums and seeds do still point at the package config: that is ")
        A("the run's *input*, and confirming it is unmodified is what those targets are for.")
        A("")
    if n[REVIEW]:
        A("### Tolerance review flags triggered")
        A("")
        A("These were **not** scored PASS or FAIL. The Stage 9 policy marks the tolerance ")
        A("`review_required`, and its preconditions were not met here:")
        A("")
        for c in checks:
            if c.status == REVIEW:
                A(f"- `{c.target_id}` (`{c.tolerance_key}`): {c.note}")
        A("")
    A("## Frozen / checksum-only artifacts")
    A("")
    A("- **Simulation dataset** — validated by sha256 only. The solver is never re-run: a chaotic ")
    A("  trajectory over ~130 Lyapunov times cannot be regenerated bit-identically, and every ")
    A("  downstream result is conditioned on these exact bytes.")
    A("- **`latent6_trial002`** — frozen checkpoint. Never retrained; a from-scratch retrain was ")
    A("  attempted and rejected (cross-environment CUDA nondeterminism). Reanalysis only.")
    A("")
    A("## Notes on selection")
    A("")
    A("- **`pinn_full64` was seed-reranked, and its physics term is now active.** The previously ")
    A("  published configuration carried `lambda_phy = 0.0`, so the physics residual was never ")
    A("  computed. The shipped configuration was rebuilt with a positive physics weight and ")
    A("  re-ranked over 10 candidates × 15 fresh seeds (9301101–9301115) = 150 validation runs ")
    A("  under the same median-based, validation-only rule, winner **C5**. Its seed-robustness ")
    A("  boxplot now ships. Report the median **19.8** with its spread (8.5–23.1). The single ")
    A("  test evaluation gave **7.3**, far below validation; that is reported as found and no ")
    A("  retuning followed. A paired ablation favours the active physics term on **validation ")
    A("  only** (19.8 vs 14.8, better on 11 of 15 seeds); there is no active-versus-zero ")
    A("  comparison on the test split, and the old and new configurations differ in too many ")
    A("  respects for their test values to isolate the physics term.")
    A("- **NGRC models are deterministic** (no RNG), so a seed check is not definable for them at all — ")
    A("  a different situation from `pinn_full64`, not the same one.")
    A("")
    (run_dir / "reproduction_report.md").write_text("\n".join(L) + "\n")


# ───────────────────────────── modes ─────────────────────────────


def cmd_list() -> None:
    pres.line("\nFinal thesis results\n")
    hdr = f"{'id':<38} {'group':<14} {'mode':<28} {'difficulty':<12} laptop-safe"
    pres.line(hdr)
    pres.line("─" * len(hdr))
    for r in REGISTRY:
        pres.line(f"{r.result_id:<38} {r.group:<14} {r.mode_label:<28} {r.difficulty:<12} "
                  f"{'yes' if r.laptop_safe else 'no'}")
    pres.line("\nFrozen artifacts — validated, never regenerated:")
    pres.line("  · simulation_dataset — the solver is never re-run; a chaotic trajectory cannot be")
    pres.line("    regenerated bit-identically, and every downstream result is conditioned on these bytes")
    pres.line("  · latent6_trial002_dynamics_baseline — frozen checkpoint; reanalysed, never retrained")
    pres.line("  · latent8_trial014_representation — FROZEN LEARNED REPRESENTATION; exact GPU retraining")
    pres.line("    is NOT claimed, so it is not offered as a reproduction path. Its checkpoint, latent")
    pres.line("    arrays and reconstruction metrics are validated against the package. Everything")
    pres.line("    downstream of it — the posthoc refit, ngrc_latent8, rc_latent8, perc_latent8,")
    pres.line("    pinn_latent8 — reproduces from it identically.")
    pres.line("\nFigure regeneration (public inputs; not part of --validate-existing model checks):")
    pres.line("  · optimizer-search figures — regenerate from the compact public dataset")
    pres.line("    forecasting/data/optimizer_search/ (32 histories, 3,040 trials) via the public plot")
    pres.line("    configs; the raw ~34 GB search trees are not shipped, but this subset is")
    pres.line("  · seed-robustness boxplots — regenerate from the shipped seed-robust provenance CSVs;")
    pres.line("    only the underlying multi-seed rerank study itself is not re-runnable from the package")
    pres.line("\nNot offered as reproduction targets, by design:")
    pres.line("  · the KSE solver — frozen; regeneration would invalidate every downstream result")
    pres.line("  · latent6 from-scratch retraining — frozen checkpoint; retrain attempted and rejected")
    pres.line("  · latent8 GPU retraining as DEFAULT reproduction — a legitimate retrain need not match")
    pres.line("    bit-for-bit across hardware, so it is never on the default path (see advanced note)")
    pres.line("\nAdvanced provenance (opt-in, NOT official reproduction):")
    pres.line("  · --advanced-gpu-latent8-historical — optional SAME-STACK latent8 historical-profile GPU")
    pres.line("    retrain (RTX 2080 Ti / torch 2.6.0 / seed 42 / deterministic). Provenance corroboration")
    pres.line("    only; never selected by --all-*; writes only under test_repro_ae_sindy/; not a cross-")
    pres.line("    hardware guarantee. Inspect with:  --plan --advanced-gpu-latent8-historical")
    pres.blank()


def plan(results: list[Result], policy: dict, targets: dict, show_commands: bool) -> None:
    pres.line("\nPlan (nothing will be executed, no run folder will be created)\n")
    for i, r in enumerate(results, 1):
        pres.line(RULE)
        pres.line(f"[{i}/{len(results)}] {r.display_name}")
        pres.line(RULE)
        pres.line(f"Mode       : {r.mode_label}   ({r.difficulty}{', GPU required' if r.requires_gpu else ''})")
        pres.line(f"Action     : {r.action}")
        if r.package_config_path:
            pres.line(f"Config     : {r.package_config_path}")
            if r.mode == "validate-only":
                pres.line(f"             -> copied verbatim to configs/{r.group}/ for reference "
                          "(nothing runs, so there is no output to redirect)")
            else:
                pres.line(f"             -> copied to configs/{r.group}/ with output redirected into the run folder")
        if r.mode == "validate-only":
            pres.line("Outputs    : validation records only (frozen artifact — nothing is recomputed)")
        else:
            pres.line(f"Outputs    : results/{r.group}/{r.result_id}/")
        pres.line(f"Plots      : {len(r.plots)} package_verified_copy" + (f" ({', '.join(r.plots)})" if r.plots else " (none ship for this result)"))
        env_line = ("not applicable — nothing is executed" if r.mode == "validate-only"
                    else "pinned deterministic profile" if r.pinned
                    else "UNPINNED — matches the validated original")
        pres.line(f"Env        : {env_line}")
        pres.line(f"Targets    : {', '.join(r.target_ids)}")
        if show_commands:
            if r.builder:
                fake = REPO_ROOT / "reproduction_runs" / "<run>" / "configs" / r.group / Path(r.package_config_path).name
                for c in BUILDERS[r.builder](r, fake):
                    pres.line(f"Command    : {' '.join(c.argv)}")
            else:
                pres.line("Command    : (none — validate-only)")
        for w in r.warnings:
            pres.line(f"⚠  {w}")
        pres.blank()

    # Surface the tolerance gates before anything runs, not after.
    keys = {targets[t]["tolerance_key"] for r in results for t in r.target_ids if t in targets}
    flagged = [k for k in sorted(keys) if policy["levels"].get(k, {}).get("review_required")]
    reran = [r for r in results if r.mode != "validate-only"]
    if reran:
        flagged.append("horizon_runtime_physical")
    if flagged:
        pres.line("Tolerance review flags in scope:")
        for k in sorted(set(flagged)):
            lvl = policy["levels"].get(k, {})
            pres.line(f"  ⚠  {k} (tolerance {lvl.get('tolerance')}) — {lvl.get('review_reason', '').strip()}")
        pres.blank()
        if reran:
            pres.line("  A fresh run promotes horizon_reporting_physical (0.05) to horizon_runtime_physical")
            pres.line("  (0.11). That band holds only under the pinned profile AND the locked seed; otherwise")
            pres.line("  the horizon comparison is reported REVIEW_REQUIRED rather than PASS/FAIL.")
            pres.blank()


def do_validate_existing(run_dir: Path, policy: dict, targets: dict) -> list[Check]:
    pres.line(f"\n{RULE}\nValidating existing package (no models are run)\n{RULE}")
    checks: list[Check] = []
    for r in REGISTRY:
        for tid in r.target_ids:
            if tid in targets:
                checks.append(validate_target(tid, targets[tid], policy, r, reran=False,
                                              observed_root=None))
    for tid in PACKAGE_TARGETS:
        if tid in targets:
            checks.append(validate_target(tid, targets[tid], policy, None, reran=False,
                                          observed_root=None))
    n = {s: sum(1 for c in checks if c.status == s) for s in (PASS, FAIL, REVIEW, SKIPPED, ERROR)}
    pres.line(f"\n{len(checks)} contract targets checked")
    pres.line(f"  PASS {n[PASS]} · FAIL {n[FAIL]} · REVIEW_REQUIRED {n[REVIEW]} · "
              f"SKIPPED {n[SKIPPED]} · ERROR {n[ERROR]}")
    for c in checks:
        if c.status in (FAIL, ERROR):
            val_logger.error("%s: %s %s — expected %s observed %s",
                             c.status, c.target_id, c.what, c.expected, c.observed)
    return checks


def do_reproduce(result: Result, idx: int, total: int, run_dir: Path, policy: dict,
                 targets: dict) -> tuple[dict, list[Check]]:
    pres.line(f"\n{RULE}\n[{idx}/{total}] {result.display_name}\n{RULE}")
    pres.line(f"Mode: {result.mode}")
    info: dict = {"commands": [], "config_notes": [], "plots": [], "status": ""}

    if result.mode == "validate-only":
        pres.line("Frozen artifact — validating checksums only; nothing is recomputed.")
        if result.package_config_path:
            src = REPO_ROOT / result.package_config_path
            dst = run_dir / "configs" / result.group / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            info["config_notes"] = [f"copied verbatim for reference (nothing runs, so no "
                                    f"output path needed redirecting): {src.name}"]
        checks = [validate_target(t, targets[t], policy, result, reran=False, observed_root=None)
                  for t in result.target_ids if t in targets]
        info["plots"] = copy_plots(result, run_dir)
        ok = all(c.status in (PASS, SKIPPED) for c in checks)
        info["status"] = "FROZEN_VALIDATED" if ok else "VALIDATION_FAILED"
        pres.line(f"Validating expected metrics ... {'PASS' if ok else 'FAIL'}")
        _print_checks(checks)
        pres.line(f"\nStatus: {info['status']}")
        return info, checks

    cfg, notes = redirect_config(result, run_dir)
    info["config_notes"] = notes
    pres.line(f"Config: {result.package_config_path}")
    pres.line(f"Output: {(run_dir / 'results' / result.group / result.result_id).relative_to(REPO_ROOT)}/")

    pre: list[str] = []
    if result.result_id == "latent6_trial002_dynamics_baseline":
        pre = stage_frozen_checkpoint(result, run_dir)
        info["config_notes"] += pre

    cmds = BUILDERS[result.builder](result, cfg)
    ok, ran = run_commands(result, cmds, run_dir)
    info["commands"] = ran
    if not ok:
        info["status"] = "REPRODUCTION_FAILED"
        checks = [Check(t, result.result_id, "reproduction", "-", "command failed", ERROR,
                        targets.get(t, {}).get("tolerance_key", "-"),
                        "result not produced; see logs/") for t in result.target_ids]
        pres.line(f"\nStatus: {info['status']}")
        return info, checks

    locked_seed, seed_note = verify_locked_seed(result, targets, run_dir)
    checks = [validate_target(t, targets[t], policy, result, reran=True, observed_root=run_dir,
                              locked_seed=locked_seed, locked_seed_note=seed_note)
              for t in result.target_ids if t in targets]
    worst = (FAIL if any(c.status == FAIL for c in checks)
             else ERROR if any(c.status == ERROR for c in checks)
             else REVIEW if any(c.status == REVIEW for c in checks) else PASS)
    pres.line(f"Validating expected metrics ... {worst}")
    _print_checks(checks)
    info["plots"] = copy_plots(result, run_dir)
    info["status"] = {PASS: "REPRODUCED_VALIDATED", REVIEW: "REPRODUCED_REVIEW_REQUIRED",
                      FAIL: "REPRODUCED_MISMATCH", ERROR: "REPRODUCED_VALIDATION_ERROR"}[worst]
    pres.line(f"\nStatus: {info['status']}")
    return info, checks


def _print_checks(checks: list[Check]) -> None:
    for c in checks:
        if c.status == SKIPPED:
            continue
        exp = c.expected if len(c.expected) <= 22 else c.expected[:19] + "…"
        obs = c.observed if len(c.observed) <= 22 else c.observed[:19] + "…"
        pres.line(f"  {c.what[:26]:<26} expected {exp:<22} observed {obs:<22} {c.status}")
        if c.status == REVIEW:
            pres.line(f"    ⚠  {c.note}")


# ───────────────────────────── interactive ─────────────────────────────


MENU_GROUPS = [
    ("Safe validation", ["simulation_dataset"]),
    ("AE-SINDy", ["latent8_trial014_representation", "latent6_trial002_dynamics_baseline",
                  "latent8_trial014_posthoc_predictive"]),
    ("Forecasting", ["ngrc_full64", "ngrc_latent8", "rc_full64", "rc_latent8",
                     "perc_full64", "perc_latent8", "pinn_full64", "pinn_latent8"]),
]


def interactive() -> tuple[list[Result], bool]:
    order = [rid for _, ids in MENU_GROUPS for rid in ids]
    pres.line("\nData-Driven Chaotic PDE Thesis Reproduction")
    pres.line("===========================================")
    pres.line("\nChoose one or more results:\n")
    num = 1
    numbering: dict[str, Result] = {}
    for title, ids in MENU_GROUPS:
        pres.line(f"{title}:")
        for rid in ids:
            r = BY_ID[rid]
            numbering[str(num)] = r
            pres.line(f"  [{num:>2}] {r.result_id:<38} {r.mode_label:<28} {r.difficulty}")
            num += 1
        pres.blank()
    pres.line("Convenience:")
    pres.line("  [A] all laptop-safe / safe validation results")
    pres.line("  [B] all forecasting selected results")
    pres.line("  [C] validate existing package only")
    pres.line("  [Q] quit")
    try:
        raw = input("\nSelect one or more, e.g. 2,5,8 or A: ").strip()
    except EOFError:
        return [], False
    if not raw or raw.upper() == "Q":
        return [], False
    if raw.upper() == "C":
        return [], True
    if raw.upper() == "A":
        chosen = [r for r in REGISTRY if r.laptop_safe]
    elif raw.upper() == "B":
        chosen = [r for r in REGISTRY if r.group == "02_forecasting"]
    else:
        chosen = []
        for tok in raw.replace(" ", "").split(","):
            if tok not in numbering:
                pres.line(f"Unknown selection: {tok}")
                return [], False
            chosen.append(numbering[tok])
    heavy = [r for r in chosen if r.difficulty == "heavy"]
    if heavy:
        pres.line("\nHeavy results selected: " + ", ".join(r.result_id for r in heavy))
        for r in heavy:
            if r.requires_gpu:
                pres.line(f"  {r.result_id}: requires a CUDA GPU; it will be skipped on a CPU-only host.")
        try:
            if input("This result may be slow on CPU-only laptops. Continue? [y/N] ").strip().lower() != "y":
                return [], False
        except EOFError:
            return [], False
    return chosen, False


# ───────────────────────────── main ─────────────────────────────


# Any tracked binary large enough to be routed through Git LFS. Used to detect a
# clone made without git-lfs, where every such file is a ~130-byte pointer stub.
_LFS_PROBE = PACKAGE / "results/00_simulation/u_series.npy"
_LFS_POINTER_PREFIX = b"version https://git-lfs"


def _rel_to_repo(path: Path) -> str:
    """Repository-relative display path, falling back to the absolute one."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def lfs_pointer_stub(path: Path) -> bool:
    """True if `path` is an unresolved Git LFS pointer rather than real content."""
    try:
        with path.open("rb") as fh:
            return fh.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX
    except OSError:
        return False


def safe_run_id(value: str) -> str:
    """Validate --run-id so a run folder cannot escape reproduction_runs/.

    pathlib joins an absolute right-hand side by REPLACING the base, and does not
    reject '..', so an unvalidated --run-id could resolve into the frozen package
    -- where stage_frozen_checkpoint() would then rmtree inside it. Mirrors the
    guard already used for output.run_group in
    forecasting/src/forecasting/runners/rc_runner.py.
    """
    if not value or value.strip() != value:
        raise ValueError("--run-id must be a non-empty value without surrounding whitespace.")
    candidate = Path(value)
    if candidate.is_absolute() or any(part in {"..", ""} for part in candidate.parts):
        raise ValueError(
            f"--run-id must be a safe relative name, got {value!r}. "
            "It must not be absolute and must not contain '..'."
        )
    if value.startswith("~"):
        raise ValueError(f"--run-id must not start with '~', got {value!r}.")
    return value


def assert_outside_package(path: Path, what: str) -> Path:
    """Refuse to write anywhere inside the frozen package.

    The package is the authoritative record; every stage writes copies of it, never
    into it. This is the last line of defence behind safe_run_id().
    """
    resolved = path.resolve()
    package = PACKAGE.resolve()
    if resolved == package or package in resolved.parents:
        raise ValueError(
            f"refusing to write {what} inside the frozen package: {resolved}\n"
            "  The package is read-only; outputs belong under reproduction_runs/."
        )
    return path


def write_run_status(run_dir: Path, verdict: str, counts: dict, *, exit_code: int,
                     note: str = "") -> None:
    """Write run_status.json atomically.

    Without it a killed or crashed run leaves a folder holding a partial results/
    tree, a run_request.json listing everything as "requested", and no record of
    what actually happened -- which reads as an authoritative result. The file is
    written via a temporary file and os.replace so it is either absent or complete,
    never half-written.
    """
    payload = {
        "verdict": verdict,
        "exit_code": exit_code,
        "counts": counts,
        "completed_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    if note:
        payload["note"] = note
    tmp = run_dir / "run_status.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, run_dir / "run_status.json")


def preflight() -> None:
    missing = [p for p in (PACKAGE, CONTRACT, TARGETS_CSV, POLICY_JSON) if not p.exists()]
    if missing:
        sys.exit("Missing required inputs:\n" + "\n".join(f"  {m}" for m in missing))
    # A clone without git-lfs satisfies the existence checks above -- every binary is
    # a pointer stub -- and then fails every checksum target with no hint why.
    if _LFS_PROBE.exists() and lfs_pointer_stub(_LFS_PROBE):
        sys.exit(
            "Git LFS content is not present: "
            f"{_rel_to_repo(_LFS_PROBE)} is an unresolved pointer file.\n"
            "  This clone was made without Git LFS, so the frozen trajectory, the model\n"
            "  checkpoints and every figure are ~130-byte placeholders. Fetch them with:\n"
            "\n      git lfs install && git lfs pull\n\n"
            "  then re-run this command."
        )


def _numeric_imports(py: str) -> bool:
    """True if interpreter `py` can import the numeric stack the pure-numpy stages need."""
    try:
        out = subprocess.run([py, "-c", "import numpy, scipy, sklearn"],
                             capture_output=True, text=True, timeout=180)
        return out.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def _torch_imports(py: str) -> bool:
    """True if interpreter `py` can 'import torch'."""
    try:
        out = subprocess.run([py, "-c", "import torch"], capture_output=True,
                             text=True, timeout=180)
        return out.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def resolve_interpreters(args: argparse.Namespace, *, require_torch: bool) -> None:
    """Resolve the system and torch interpreters and set the SYS_PY / CONDA_PY globals.

    Precedence — system interpreter (pure numpy/scipy steps):
      1. --system-python
      2. $REPRO_SYSTEM_PYTHON
      3. the original HPC system interpreter, if it exists on this machine
      4. sys.executable (the interpreter running this script)

    Precedence — torch interpreter (latent decode, PINN, AE-SINDy):
      1. --torch-python
      2. $REPRO_TORCH_PYTHON
      3. the original HPC conda interpreter, if it exists on this machine
      4. sys.executable, if it can import torch
      5. otherwise a clear error explaining how to point at a torch interpreter

    One interpreter (e.g. a single project .venv with torch) may fill both roles.
    """
    global SYS_PY, CONDA_PY

    # --- system interpreter ---
    # /usr/bin/python3 exists on essentially every Linux host, so preferring it
    # unconditionally silently bypassed the user's venv -- and a distro numpy of a
    # different version would produce unvalidated numbers. It is now used only if it
    # is explicitly requested, or if the active interpreter cannot satisfy the
    # numeric stack while it can.
    sys_py = (args.system_python
              or os.environ.get("REPRO_SYSTEM_PYTHON")
              or (sys.executable if _numeric_imports(sys.executable) else None)
              or (_HPC_SYS_PY if Path(_HPC_SYS_PY).exists()
                  and _numeric_imports(_HPC_SYS_PY) else None)
              or sys.executable)
    if not Path(sys_py).exists():
        sys.exit(f"System interpreter not found: {sys_py}\n"
                 "  Set one with --system-python PATH or $REPRO_SYSTEM_PYTHON.")

    # --- torch interpreter ---
    torch_py = (args.torch_python
                or os.environ.get("REPRO_TORCH_PYTHON")
                or (_HPC_CONDA_PY if Path(_HPC_CONDA_PY).exists() else None))
    if torch_py is None:
        if _torch_imports(sys.executable):
            torch_py = sys.executable
        elif not require_torch:
            # This command runs no torch (e.g. --plan / --validate-existing); keep a
            # usable path for display without failing.
            torch_py = sys_py
        else:
            sys.exit(
                "No torch-capable interpreter found.\n"
                "  The default reproduction decodes latents and runs the PINN and "
                "AE-SINDy models, which require PyTorch.\n"
                "  Point at an interpreter that can 'import torch' with --torch-python "
                "PATH or $REPRO_TORCH_PYTHON\n"
                "  (for example a virtual environment created from requirements.txt).")
    if not Path(torch_py).exists():
        sys.exit(f"Torch interpreter not found: {torch_py}\n"
                 "  Set one with --torch-python PATH or $REPRO_TORCH_PYTHON.")
    if require_torch and not _torch_imports(torch_py):
        sys.exit(f"Selected torch interpreter cannot import torch: {torch_py}\n"
                 "  Install PyTorch there, or point at a different interpreter with "
                 "--torch-python PATH or $REPRO_TORCH_PYTHON.")

    SYS_PY = sys_py
    CONDA_PY = torch_py
    logger.debug("resolved system interpreter: %s", SYS_PY)
    logger.debug("resolved torch interpreter : %s", CONDA_PY)


def gpu_available() -> bool:
    try:
        out = subprocess.run([CONDA_PY, "-c", "import torch;print(torch.cuda.is_available())"],
                             capture_output=True, text=True, timeout=120)
        return out.stdout.strip() == "True"
    except Exception:  # noqa: BLE001
        return False


# ───────────────── advanced provenance path (opt-in, NON-DEFAULT) ─────────────────
#
# latent8_trial014_representation is FROZEN / checksum-validated by default (2026-07-17
# policy lock) and is NEVER retrained by the default reproduction. This opt-in path exposes
# the historical GPU retrain PURELY as same-stack provenance corroboration. It is:
#   · never selected by --all-safe / --all-forecasting / interactive default selection,
#   · never mixed into normal validation (it is not a REGISTRY entry),
#   · guaranteed to write ONLY under test_repro_ae_sindy/ — never into the package,
#   · a same-stack observation only — NOT a cross-hardware GPU guarantee.
# A match is corroboration; a non-match invalidates nothing.
ADV_SCRATCH_ROOT = REPO_ROOT / "test_repro_ae_sindy"
ADV_CONFIG_SRC = "final_thesis_package_v001/configs/01_ae_sindy/latent8_trial014_representation.json"
ADV_DATA_PATH = "final_thesis_package_v001/results/00_simulation/u_series.npy"
ADV_DATA_SHA = "85bb1902875e2b3062685cf327f4695e9e106bfbae7b39b70961e92b520a4417"
ADV_MODEL_SHA = "6e6c38b5d06e9b5f22944210a87bd3236b7705a38e7bd52dcafe680a72d1cc58"
ADV_MSE_MEAN = "0.006592637859284878"
ADV_MAE_MEAN = "0.062012095004320145"
ADV_RUN_NAME = "latent8_trial014_historical_profile_test"
ADV_PROFILE = [  # (label, expected)
    ("gpu_name contains", "NVIDIA GeForce RTX 2080 Ti"),
    ("python executable", CONDA_PY),
    ("python version", "3.12.10"),
    ("torch", "2.6.0"),
    ("torch.version.cuda", "12.6"),
    ("cudnn", "90800"),
    ("seed", "42"),
    ("deterministic", "true"),
    ("u_series.npy sha256", ADV_DATA_SHA),
]


def _adv_env_probe() -> dict:
    """Read-only, best-effort probe of the conda interpreter. Imports torch to read version
    strings; never runs a model, never does GPU compute."""
    info = {"conda_py_exists": Path(CONDA_PY).exists()}
    try:
        code = ("import platform,json,torch;"
                "print(json.dumps({'python':platform.python_version(),"
                "'torch':torch.__version__,'cuda':torch.version.cuda,"
                "'cudnn':str(torch.backends.cudnn.version()),"
                "'cuda_available':torch.cuda.is_available(),"
                "'gpu':(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')}))")
        out = subprocess.run([CONDA_PY, "-c", code], capture_output=True, text=True, timeout=180)
        info.update(json.loads((out.stdout or "").strip() or "{}"))
    except Exception:  # noqa: BLE001
        pass
    return info


def _adv_data_ok() -> bool:
    p = REPO_ROOT / ADV_DATA_PATH
    try:
        return p.exists() and sha256(p) == ADV_DATA_SHA
    except Exception:  # noqa: BLE001
        return False


def _adv_paths(scratch: Path) -> dict[str, Path]:
    return {
        "cfg": scratch / "configs" / "latent8_trial014_representation_historical_test.json",
        "run": scratch / "results" / ADV_RUN_NAME,
        "lat": scratch / "results" / "latent_data",
        "sim": REPO_ROOT / ADV_DATA_PATH,
    }


def _adv_commands(scratch: Path) -> list[str]:
    q = _adv_paths(scratch)
    return [
        f"{CONDA_PY} ae_sindy/scripts/run_train.py --config {q['cfg']}",
        f"{CONDA_PY} ae_sindy/scripts/run_analysis.py --config {q['cfg']}",
        f"{CONDA_PY} ae_sindy/scripts/run_visualize.py --config {q['cfg']}",
        (f"{CONDA_PY} scripts/export_final_reproduction_latents.py --run-dir {q['run']} "
         f"--source-config {q['cfg']} --output-dir {q['lat']} --label {ADV_RUN_NAME} "
         f"--data-path {q['sim']} --train-end 20000 --valid-end 25000 --device auto"),
    ]


def _adv_profile_ok(probe: dict) -> bool:
    return bool(
        probe.get("conda_py_exists")
        and probe.get("cuda_available") is True
        and "RTX 2080 Ti" in (probe.get("gpu") or "")
        and probe.get("torch") == "2.6.0"
        and str(probe.get("cuda")) == "12.6"
        and str(probe.get("cudnn")) == "90800"
        and (probe.get("python") or "").startswith("3.12.10")
        and _adv_data_ok()
    )


def advanced_plan(scratch_label: str = "<timestamp>") -> None:
    """Inspection only. Prints the advanced provenance plan; creates nothing, runs nothing."""
    scratch = ADV_SCRATCH_ROOT / f"latent8_trial014_historical_profile_{scratch_label}"
    probe = _adv_env_probe()
    observed = {
        "python version": probe.get("python"),
        "torch": probe.get("torch"),
        "torch.version.cuda": probe.get("cuda"),
        "cudnn": probe.get("cudnn"),
        "python executable": CONDA_PY if probe.get("conda_py_exists") else f"MISSING ({CONDA_PY})",
        "gpu_name contains": probe.get("gpu") or "(no GPU visible on this node)",
    }
    pres.line("\n" + RULE)
    pres.line("ADVANCED PROVENANCE PATH — AE-SINDy latent8 historical-profile GPU retrain")
    pres.line(RULE)
    pres.line("⚠  This is an optional SAME-STACK provenance retrain, not the official portable")
    pres.line("   reproduction path. Opt-in only; never selected by --all-safe/--all-forecasting;")
    pres.line("   never mixed into normal validation. The default result stays FROZEN/checksum-")
    pres.line("   validated. A match corroborates; a non-match invalidates nothing.")
    pres.line("⚠  NOT a cross-hardware GPU guarantee: on different hardware a legitimate retrain")
    pres.line("   need not match bit-for-bit (the nondeterminism that rejected the latent6 retrain).")
    pres.blank()
    pres.line("Required historical profile (expected — observed where checkable read-only):")
    for label, expected in ADV_PROFILE:
        obs = observed.get(label)
        if obs is None or label in ("seed", "deterministic", "u_series.npy sha256"):
            extra = ""
        elif str(expected) in str(obs) or str(obs) == str(expected):
            extra = "   ✓"
        else:
            extra = f"   observed: {obs}"
        pres.line(f"  · {label:<22} {expected}{extra}")
    pres.line(f"  · {'u_series.npy present':<22} "
              + ("✓ present & checksum matches" if _adv_data_ok() else "NOT verified on this host"))
    pres.blank()
    pres.line("Scratch output root (created only on execute; NOTHING writes to the package):")
    pres.line(f"  {scratch.relative_to(REPO_ROOT)}/  -> configs/ logs/ results/ reports/ scripts/")
    pres.blank()
    pres.line("Exact commands that would run — only if explicitly executed on a matching GPU node:")
    for c in _adv_commands(scratch):
        pres.line(f"  {c}")
    pres.blank()
    pres.line("Comparison targets (package artifacts to match):")
    pres.line(f"  · model.pt sha256          {ADV_MODEL_SHA}")
    pres.line(f"  · reconstruction MSE mean  {ADV_MSE_MEAN}")
    pres.line(f"  · reconstruction MAE mean  {ADV_MAE_MEAN}")
    pres.line("  · latent arrays z_series / z_train / z_valid / z_test / u_reconstructed (byte-level)")
    pres.line(RULE)
    pres.line("Inspection only — nothing was executed and no run folder was created.")
    pres.line("To run it (opt-in): re-run without --plan on a matching RTX 2080 Ti node, then --yes")
    pres.line("to build the scratch folder and submit ONE detached GPU job.\n")


def _adv_write_scaffold(scratch: Path) -> Path:
    """Materialize the copied+redirected config and a Slurm submit script under the scratch
    folder. Writes ONLY under test_repro_ae_sindy/. Returns the Slurm script path."""
    q = _adv_paths(scratch)
    for sub in ("configs", "logs", "results", "reports", "scripts"):
        (scratch / sub).mkdir(parents=True, exist_ok=True)
    cfg = json.loads((REPO_ROOT / ADV_CONFIG_SRC).read_text())
    cfg["output_dir"] = str(scratch / "results")
    cfg["run_name"] = ADV_RUN_NAME
    cfg["data_file"] = ADV_DATA_PATH  # package frozen simulation (read)
    q["cfg"].write_text(json.dumps(cfg, indent=2) + "\n")
    slurm = scratch / "scripts" / "submit_latent8_historical_test.slurm"
    logdir = scratch / "logs"
    body = "\n".join(_adv_commands(scratch))
    slurm.write_text(
        "#!/bin/bash\n"
        "#SBATCH --job-name=l8_hist_test\n"
        "#SBATCH --partition=work\n#SBATCH --account=dsnf\n#SBATCH --qos=normal\n"
        "#SBATCH --time=04:00:00\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n"
        "#SBATCH --cpus-per-task=8\n#SBATCH --gres=gpu:rtx2080ti:1\n"
        f"#SBATCH --output={logdir}/hist_test_%j.out\n"
        f"#SBATCH --error={logdir}/hist_test_%j.err\n\n"
        "set -eo pipefail\n"
        "# EXPLORATORY same-stack provenance retrain — NOT official reproduction. Writes only\n"
        "# under test_repro_ae_sindy/. gpu pinned to rtx2080ti; repro_env.sh deliberately NOT sourced.\n"
        f"cd {REPO_ROOT}\nexport PYTHONPATH={REPO_ROOT}/ae_sindy:${{PYTHONPATH:-}}\n"
        'echo "node: $(hostname)"; nvidia-smi || true\n'
        f'{CONDA_PY} - <<PYEOF\nimport hashlib\nh=hashlib.sha256(open("{q["sim"]}","rb").read()).hexdigest()\n'
        f'assert h=="{ADV_DATA_SHA}", "input checksum mismatch"\nprint("u_series match_locked: True")\nPYEOF\n'
        f"{body}\n"
        f'{CONDA_PY} - <<PYEOF\nimport hashlib\ng=hashlib.sha256(open("{q["run"]}/model.pt","rb").read()).hexdigest()\n'
        f'print("model.pt this run:", g)\nprint("MODEL_BIT_IDENTICAL:", g=="{ADV_MODEL_SHA}")\nPYEOF\n'
    )
    slurm.chmod(0o755)
    return slurm


def advanced_execute(assume_yes: bool, force: bool) -> int:
    advanced_plan()
    probe = _adv_env_probe()
    ok = _adv_profile_ok(probe)
    if not ok and not force:
        pres.line("✗ Historical profile NOT satisfied on this host (need an RTX 2080 Ti node with the")
        pres.line("  pinned conda torch 2.6.0 / CUDA 12.6 / cuDNN 9.8.0 stack). Refusing to run.")
        pres.line("  Inspect with --plan, run on a matching GPU node, or pass")
        pres.line("  --force-advanced-profile to override (not recommended; result would not be comparable).")
        return 2
    if not assume_yes:
        pres.line("Profile satisfied." if ok else "Profile OVERRIDDEN via --force-advanced-profile.")
        pres.line("This would build a scratch run folder and submit ONE detached GPU job.")
        pres.line("Re-run with --yes to build and submit. Nothing was written.")
        return 0
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    scratch = ADV_SCRATCH_ROOT / f"latent8_trial014_historical_profile_{ts}"
    slurm = _adv_write_scaffold(scratch)
    pres.line(f"\nScaffold written under {scratch.relative_to(REPO_ROOT)}/ (package untouched).")
    if shutil.which("sbatch"):
        res = subprocess.run(["sbatch", str(slurm)], capture_output=True, text=True)
        pres.line((res.stdout or "").strip() or (res.stderr or "").strip())
        return 0 if res.returncode == 0 else 1
    pres.line(f"sbatch not found — submit manually:\n  sbatch {slurm}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="reproduce_final_thesis.py",
        description="Reproduce and validate the final thesis results against the "
                    "validation contract in repro_validation/.",
        epilog="Frozen by design: the KSE solver is never re-run and latent6_trial002 is "
               "never retrained. Historical optimiser searches and seed-reranking experiments "
               "are not rerun by this script. Their released diagnostic figures can be "
               "regenerated separately from the compact public data included in the repository.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="list the 12 final results and exit")
    p.add_argument("--results", nargs="+", metavar="ID", help="result ids to reproduce")
    p.add_argument("--all-safe", action="store_true", help="all laptop-safe results")
    p.add_argument("--all-forecasting", action="store_true", help="all 8 forecasting results")
    p.add_argument("--all-results", action="store_true",
                   help="all 12 final results (the documented default set)")
    p.add_argument("--plan", action="store_true", help="show what would happen; create nothing")
    p.add_argument("--dry-run", action="store_true", help="like --plan, and print the exact commands")
    p.add_argument("--validate-existing", action="store_true",
                   help="validate the existing package against the contract; runs no models")
    p.add_argument("--run-id", metavar="NAME", help="run folder name (default: timestamp)")
    p.add_argument("--yes", action="store_true", help="skip confirmation for heavy results")
    p.add_argument("--advanced-gpu-latent8-historical", action="store_true",
                   help="OPT-IN, NON-DEFAULT: inspect/run the same-stack latent8 historical-profile "
                        "GPU retrain (provenance only; not official reproduction; writes only under "
                        "test_repro_ae_sindy/). Use with --plan to inspect without running.")
    p.add_argument("--force-advanced-profile", action="store_true",
                   help="override the RTX 2080 Ti profile guard for the advanced path (not recommended)")
    p.add_argument("--system-python", metavar="PATH",
                   help="interpreter for pure numpy/scipy steps (full-64 forecasting). "
                        "Default: $REPRO_SYSTEM_PYTHON, else the original HPC path if it "
                        "exists, else this interpreter.")
    p.add_argument("--torch-python", metavar="PATH",
                   help="interpreter with PyTorch (latent decode, PINN, AE-SINDy). "
                        "Default: $REPRO_TORCH_PYTHON, else the original HPC conda path if "
                        "it exists, else this interpreter when it can import torch. "
                        "A single .venv can serve both roles.")
    add_logging_arguments(p)
    args = p.parse_args(argv)

    # One authoritative logging policy for this process (component REPRO). The
    # child forecasting/AE-SINDy runs configure their own logging; only an
    # explicit --verbose/--debug/--log-level is propagated to them (via env),
    # so a default run leaves child behaviour exactly as before.
    log_settings = configure_from_args(args, component="REPRO")
    if args.verbose or args.debug or args.log_level:
        os.environ.update(log_settings.as_child_env())
    logger.debug("logging level=%s verbose=%s debug=%s",
                 log_settings.level, log_settings.verbose, log_settings.debug)

    preflight()

    if args.list:
        cmd_list()
        return 0

    # Resolve the system/torch interpreters (portable venv or original HPC paths).
    # torch import is enforced only when models will actually run.
    require_torch = not (args.plan or args.dry_run or args.validate_existing)
    resolve_interpreters(args, require_torch=require_torch)

    if args.advanced_gpu_latent8_historical:
        if args.plan or args.dry_run:
            advanced_plan()
            return 0
        return advanced_execute(assume_yes=args.yes, force=args.force_advanced_profile)

    policy, targets = load_policy(), load_targets()

    selected: list[Result] = []
    validate_existing = args.validate_existing
    if args.results:
        unknown = [r for r in args.results if r not in BY_ID]
        if unknown:
            logger.error("Unknown result id(s): %s. Use --list to see valid ids.",
                         ", ".join(unknown))
            return 2
        selected = [BY_ID[r] for r in args.results]
    elif args.all_safe:
        selected = [r for r in REGISTRY if r.laptop_safe]
    elif args.all_results:
        selected = list(REGISTRY)
    elif args.all_forecasting:
        selected = [r for r in REGISTRY if r.group == "02_forecasting"]
    elif not validate_existing:
        selected, validate_existing = interactive()
        if not selected and not validate_existing:
            pres.line("Nothing selected.")
            return 0
        args.yes = True  # the menu already confirmed

    if args.plan or args.dry_run:
        if validate_existing:
            pres.line("\nPlan: validate the existing package against the Stage 9 contract.")
            pres.line(f"  {len(targets)} targets would be checked; no models would run; no package file written.")
            return 0
        plan(selected, policy, targets, show_commands=args.dry_run)
        return 0

    if not validate_existing and not args.yes:
        heavy = [r for r in selected if r.difficulty == "heavy"]
        if heavy:
            logger.error("Refusing to start heavy results without --yes: %s",
                         ", ".join(r.result_id for r in heavy))
            logger.error("Re-run with --yes, or use --plan/--dry-run to preview.")
            return 2

    try:
        run_id = safe_run_id(args.run_id) if args.run_id \
            else _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    run_dir = RUNS_ROOT / run_id
    try:
        assert_outside_package(run_dir, "the run folder")
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("logs", "configs", "results", "plots", "validation"):
        (run_dir / sub).mkdir(exist_ok=True)

    env_notes: list[str] = []
    drift = [k for k, v in PINNED_ENV.items() if os.environ.get(k) != v]
    if drift:
        env_notes.append("The launching shell did not have the deterministic profile set "
                         f"({', '.join(drift)}); it was applied to child processes instead of "
                         "running unpinned.")
        logger.info("Applying the deterministic profile (OMP_NUM_THREADS=1 …) to child "
                    "processes; the launching shell did not have it set.")

    # A GPU-gated result on a CPU-only host is downgraded rather than run wrong.
    kept: list[Result] = []
    for r in selected:
        if r.requires_gpu and not validate_existing:
            if not gpu_available():
                logger.warning("%s requires a CUDA GPU and none is available; "
                               "downgrading it to validate-only rather than running it incorrectly.",
                               r.result_id)
                r = Result(**{**r.__dict__, "mode": "validate-only", "builder": None})
                env_notes.append(f"{r.result_id}: no GPU available — validated only, not reproduced.")
        kept.append(r)
    selected = kept

    request = {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "mode": "validate-existing" if validate_existing else "reproduce",
        "requested_results": [r.result_id for r in selected],
        "contract": {
            "targets": str(TARGETS_CSV.relative_to(REPO_ROOT)),
            "policy": str(POLICY_JSON.relative_to(REPO_ROOT)),
        },
        "package_is_read_only": True,
        "solver_rerun": False,
        "latent6_retrained": False,
    }
    (run_dir / "run_request.json").write_text(json.dumps(request, indent=2) + "\n")

    checks: list[Check] = []
    per_result: dict = {}
    if validate_existing:
        checks = do_validate_existing(run_dir, policy, targets)
    else:
        for i, r in enumerate(selected, 1):
            info, cs = do_reproduce(r, i, len(selected), run_dir, policy, targets)
            per_result[r.result_id] = info
            checks.extend(cs)

    pinned_applied = any(r.pinned for r in selected) or validate_existing
    (run_dir / "environment_snapshot.txt").write_text(environment_snapshot(pinned_applied, env_notes))
    write_validation(run_dir, checks)
    write_report(run_dir, request, selected, checks, per_result, env_notes, validate_existing)

    n = {s: sum(1 for c in checks if c.status == s) for s in (PASS, FAIL, REVIEW, SKIPPED, ERROR)}
    logger.debug("run folder: %s", run_dir.relative_to(REPO_ROOT))
    logger.debug("report: %s", (run_dir / "reproduction_report.md").relative_to(REPO_ROOT))
    pres.line(f"\n{RULE}")
    pres.line(f"Run folder : {run_dir.relative_to(REPO_ROOT)}/")
    pres.line(f"Report     : {(run_dir / 'reproduction_report.md').relative_to(REPO_ROOT)}")
    pres.line(f"Validation : {n[PASS]} PASS · {n[FAIL]} FAIL · {n[REVIEW]} REVIEW_REQUIRED · "
              f"{n[SKIPPED]} SKIPPED · {n[ERROR]} ERROR")

    if n[FAIL] or n[ERROR]:
        verdict, code = "FAILED", 1
    elif n[REVIEW]:
        verdict, code = "REVIEW_REQUIRED", 3
    else:
        verdict, code = "PASSED", 0
    write_run_status(run_dir, verdict, dict(n), exit_code=code)
    pres.line(f"Verdict    : {verdict}  (exit {code})")
    pres.line(RULE)
    return code


if __name__ == "__main__":
    sys.exit(main())
