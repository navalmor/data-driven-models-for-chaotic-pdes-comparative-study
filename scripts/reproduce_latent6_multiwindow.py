from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _bootstrap_path in (_REPO_ROOT, _REPO_ROOT / "ae_sindy"):
    if str(_bootstrap_path) not in sys.path:
        sys.path.insert(0, str(_bootstrap_path))

import numpy as np
import torch

from common.logging_setup import add_logging_arguments, configure_from_args, get_logger
from ae_sindy.analysis import load_config_from_run, run_analysis
from ae_sindy.data_utils import load_array, load_normalization_stats, prepare_kse_datasets

logger = get_logger("latent6_multiwindow")


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "final_thesis_package_v001"


def assert_outside_package(path: Path) -> Path:
    """Refuse to write inside the frozen package.

    The eight files this script produces are covered by
    stage4a_latent6_trial002_checkpoint_reanalysis_sha256_manifest.csv and are cited
    by contract target AE6-002. Writing there would overwrite authoritative artefacts
    without verifying anything.
    """
    resolved = Path(path).expanduser().resolve()
    package = PACKAGE.resolve()
    if resolved == package or package in resolved.parents:
        raise SystemExit(
            f"--output-dir must resolve outside the frozen package, got {resolved}.\n"
            "  The packaged multi-window evaluation is authoritative and manifest-covered;\n"
            "  write to a scratch directory instead."
        )
    return resolved


RUN_DIR = Path(
    "final_thesis_package_v001/results/01_ae_sindy/"
    "latent6_trial002_dynamics_baseline/latent6_trial002_dynamics_baseline"
)
CONFIG_PATH = Path(
    "final_thesis_package_v001/configs/01_ae_sindy/"
    "latent6_trial002_dynamics_baseline_checkpoint_reanalysis.json"
)
# There is deliberately NO default output directory inside the package. The eight
# files this script writes are covered by
# stage4a_latent6_trial002_checkpoint_reanalysis_sha256_manifest.csv and their numbers
# are cited by contract target AE6-002, so writing there by default would let a rerun
# silently overwrite authoritative artefacts without verifying anything. --output-dir
# is required and must resolve outside the package.
DEFAULT_OUTPUT_DIR = None

# Held-out report windows, identical to scripts/posthoc_refit_sindy.py::TEST_WINDOWS.
# They are fixed in advance and are never used to select a model or a threshold.
TEST_WINDOWS: List[Tuple[int, int]] = [
    (25000, 26000),
    (26000, 27000),
    (27000, 28000),
    (28000, 29000),
    (29000, 30000),
]

EXPECTED_CHECKPOINT_SHA256 = "30c3f2e16039142e6f82939c0ebc469af6238a8d24a1686c6bfbe4c8301d8294"
EXPECTED_LATENT_DIM = 6
EXPECTED_LIBRARY_DIM = 28

INPUT_FILES = ("model.pt", "params.pkl", "config.json", "normalization_stats.json")

DETAIL_FIELDS = [
    "window_index",
    "window_start",
    "window_end",
    "n_samples",
    "physical_start_time",
    "physical_end_time",
    "sindy_valid_time",
    "sindy_first_invalid_time",
    "sindy_valid_index",
    "sindy_first_invalid_index",
    "sindy_threshold_crossed",
    "right_censored",
    "encoder_valid_time",
    "encoder_rollout_valid_time",
    "decoder_error",
    "active_terms",
    "sparsity_percent",
    "error_threshold",
    "solver_success",
    "rollout_success",
    "used_fallback",
    "requested_steps",
    "used_steps",
    "rollout_status",
    "finite_outputs",
    "max_finite_sindy_error",
    "sindy_mean_error",
    "sindy_max_error",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_inputs(run_dir: Path, config) -> Dict[str, str]:
    checksums = {}
    for name in INPUT_FILES:
        path = run_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen input {path}")
        checksums[name] = sha256(path)

    if checksums["model.pt"] != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"Checkpoint sha256 mismatch for {run_dir / 'model.pt'}: "
            f"expected {EXPECTED_CHECKPOINT_SHA256}, got {checksums['model.pt']}"
        )
    if config.latent_dim != EXPECTED_LATENT_DIM:
        raise ValueError(f"Expected latent_dim={EXPECTED_LATENT_DIM}, got {config.latent_dim}")
    if config.library_dim != EXPECTED_LIBRARY_DIM:
        raise ValueError(f"Expected library_dim={EXPECTED_LIBRARY_DIM}, got {config.library_dim}")

    return checksums


def evaluate_window(
    config,
    series: np.ndarray,
    window: Tuple[int, int],
    window_index: int,
    run_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    stats = load_normalization_stats(run_dir / "normalization_stats.json") if config.normalize else None

    _, _, test_data, _ = prepare_kse_datasets(
        series,
        dt=config.dt,
        model_order=config.model_order,
        train_slice=None,
        val_slice=None,
        test_slice=window,
        normalize=config.normalize,
        normalization_stats=stats,
    )
    if test_data is None:
        raise ValueError(f"Window {window} produced no test data")

    expected_samples = window[1] - window[0]
    if test_data["x"].shape[0] != expected_samples:
        raise ValueError(
            f"Window {window} produced {test_data['x'].shape[0]} samples, expected {expected_samples}"
        )
    if test_data["x"].shape[1] != config.input_dim:
        raise ValueError(
            f"Window {window} produced width {test_data['x'].shape[1]}, expected {config.input_dim}"
        )

    # run_analysis writes analysis_summary.pkl beside the checkpoint it loads, so it is
    # given a staging copy and the canonical run directory is never written to.
    with tempfile.TemporaryDirectory(prefix="latent6_multiwindow_") as staging:
        staging_dir = Path(staging)
        for name in INPUT_FILES:
            shutil.copy2(run_dir / name, staging_dir / name)

        summary = run_analysis(
            staging_dir,
            test_data,
            dt=config.dt,
            device="cpu",
            generate_plots=False,
            rollout_max_steps=config.analysis_rollout_max_steps,
        )

        # The staging directory is ephemeral; record the frozen checkpoint that was
        # actually loaded, verified by checksum above.
        summary["checkpoint_path"] = str(run_dir / "model.pt")

        window_dir = output_dir / f"window_{window_index:02d}_{window[0]}_{window[1]}"
        window_dir.mkdir(parents=True, exist_ok=True)
        with (window_dir / "analysis_summary.pkl").open("wb") as f:
            pickle.dump(summary, f)

    meta = summary.get("rollout_meta", {})
    errors = summary.get("sindy_error_over_time")
    if errors is None:
        max_finite_error = None
        finite_outputs = None
    else:
        errors = np.asarray(errors, dtype=np.float64)
        finite = np.isfinite(errors)
        max_finite_error = float(np.max(errors[finite])) if finite.any() else None
        finite_outputs = bool(finite.all())

    crossed = summary.get("sindy_threshold_crossed")
    logger.info(
        "window %d [%d,%d) | horizon=%s crossed=%s solver_success=%s status=%s",
        window_index,
        window[0],
        window[1],
        summary.get("sindy_valid_time"),
        crossed,
        meta.get("solver_success"),
        meta.get("rollout_status"),
    )

    return {
        "window_index": window_index,
        "window_start": window[0],
        "window_end": window[1],
        "n_samples": int(test_data["x"].shape[0]),
        "physical_start_time": window[0] * config.dt,
        "physical_end_time": (window[1] - 1) * config.dt,
        "sindy_valid_time": summary.get("sindy_valid_time"),
        "sindy_first_invalid_time": summary.get("sindy_first_invalid_time"),
        "sindy_valid_index": summary.get("sindy_valid_index"),
        "sindy_first_invalid_index": summary.get("sindy_first_invalid_index"),
        "sindy_threshold_crossed": crossed,
        "right_censored": (crossed is False),
        "encoder_valid_time": summary.get("encoder_valid_time"),
        "encoder_rollout_valid_time": summary.get("encoder_rollout_valid_time"),
        "decoder_error": summary.get("decoder_error"),
        "active_terms": summary.get("active_terms"),
        "sparsity_percent": summary.get("sparsity_percent"),
        "error_threshold": summary.get("horizon_error_threshold"),
        "solver_success": meta.get("solver_success"),
        "rollout_success": meta.get("rollout_success"),
        "used_fallback": meta.get("used_fallback"),
        "requested_steps": meta.get("requested_steps"),
        "used_steps": meta.get("used_steps"),
        "rollout_status": meta.get("rollout_status"),
        "finite_outputs": finite_outputs,
        "max_finite_sindy_error": max_finite_error,
        "sindy_mean_error": summary.get("sindy_mean_error"),
        "sindy_max_error": summary.get("sindy_max_error"),
    }


def write_protocol(
    path: Path,
    config,
    run_dir: Path,
    checksums: Dict[str, str],
    series_path: Path,
    series: np.ndarray,
    stats,
) -> None:
    protocol = {
        "purpose": (
            "Structured multi-window test evaluation of the frozen jointly trained "
            "latent-6 AE-SINDy model."
        ),
        "evaluation_type": "analysis_only_no_refit_no_retrain_no_model_selection",
        "supports_result": "jointly trained latent-6 AE-SINDy (structural result)",
        "model_identity": "latent6_trial002_dynamics_baseline",
        "run_dir": str(run_dir),
        "config_path": str(CONFIG_PATH),
        "checkpoint_path": str(run_dir / "model.pt"),
        "checkpoint_sha256": checksums["model.pt"],
        "params_pkl_sha256": checksums["params.pkl"],
        "run_config_json_sha256": checksums["config.json"],
        "normalization_stats_sha256": checksums["normalization_stats.json"],
        "normalization_mean": stats.mean,
        "normalization_std": stats.std,
        "normalization_kind": "global scalar z-score, train-only fit",
        "simulation_data_path": str(series_path),
        "simulation_data_sha256": sha256(series_path),
        "simulation_shape": list(series.shape),
        "train_slice": list(config.train_slice),
        "val_slice": list(config.val_slice),
        "canonical_test_slice": list(config.test_slice),
        "windows": [list(w) for w in TEST_WINDOWS],
        "windows_source": "scripts/posthoc_refit_sindy.py::TEST_WINDOWS",
        "windows_role": "test_report",
        "windows_fixed_before_evaluation": True,
        "test_used_for_selection": False,
        "latent_dim": config.latent_dim,
        "poly_order": config.poly_order,
        "library_dim": config.library_dim,
        "model_order": config.model_order,
        "activation": config.activation,
        "widths": list(config.widths),
        "dt": config.dt,
        "solver": "RK45",
        "solver_rtol": 1e-8,
        "solver_atol": 1e-10,
        "solver_max_step": config.dt / 2.0,
        "rollout_max_steps": config.analysis_rollout_max_steps,
        "error_definition": (
            "per-timestep relative L2: ||u_true - u_pred||_2 / (||u_true||_2 + 1e-12), "
            "normalized field"
        ),
        "error_threshold": config.horizon_error_threshold,
        "horizon_convention": "last_valid (valid_index = first_invalid_index - 1)",
        "time_reporting": "physical simulation time (dt=0.1), rollout time axis re-based to 0",
        "lambda_max": config.lambda_max,
        "std_convention": "population standard deviation, ddof=0",
        "coefficients_source": (
            "frozen checkpoint sindy_coefficients * frozen params.pkl coefficient_mask"
        ),
        "model_modified": False,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the frozen latent-6 five-window held-out test evaluation "
            "(analysis only; no training, refit or model selection)."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=RUN_DIR,
        help="Frozen latent-6 run directory holding model.pt and the run configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination for the protocol, per-window summaries and the two CSV tables. "
             "Required, and must resolve outside final_thesis_package_v001/.",
    )
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_from_args(args, component="SINDy")

    assert_outside_package(args.output_dir)

    run_dir = args.run_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config_from_run(run_dir)
    checksums = check_inputs(run_dir, config)
    logger.info("frozen checkpoint verified | sha256=%s", checksums["model.pt"])

    series_path = Path(config.data_file)
    series = load_array(series_path)
    stats = load_normalization_stats(run_dir / "normalization_stats.json")

    rows = [
        evaluate_window(config, series, window, index, run_dir, output_dir)
        for index, window in enumerate(TEST_WINDOWS)
    ]

    detail_path = output_dir / "latent6_multiwindow_detail.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    horizons = [float(row["sindy_valid_time"]) for row in rows]
    summary = {
        "model": "latent6_trial002_dynamics_baseline",
        "evaluation": "frozen latent-6 five-window test evaluation",
        "n_windows": len(rows),
        "mean_sindy_valid_time": float(np.mean(horizons)),
        "min_sindy_valid_time": float(np.min(horizons)),
        "max_sindy_valid_time": float(np.max(horizons)),
        "std_sindy_valid_time": float(np.std(horizons, ddof=0)),
        "horizon_window_00": horizons[0],
        "horizon_window_01": horizons[1],
        "horizon_window_02": horizons[2],
        "horizon_window_03": horizons[3],
        "horizon_window_04": horizons[4],
        "threshold_crossed_count": int(sum(1 for row in rows if row["sindy_threshold_crossed"])),
        "right_censored_count": int(sum(1 for row in rows if row["right_censored"])),
        "solver_success_count": int(sum(1 for row in rows if row["solver_success"])),
        "mean_decoder_error": float(np.mean([row["decoder_error"] for row in rows])),
        "active_terms": rows[0]["active_terms"],
        "sparsity_percent": rows[0]["sparsity_percent"],
        "error_threshold": rows[0]["error_threshold"],
        "time_units": "physical simulation time",
        "std_convention": "ddof=0",
        "checkpoint_sha256": checksums["model.pt"],
    }
    summary_path = output_dir / "latent6_multiwindow_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    write_protocol(
        output_dir / "latent6_multiwindow_protocol.json",
        config,
        run_dir,
        checksums,
        series_path,
        series,
        stats,
    )

    logger.info(
        "horizons=%s mean=%.6f min=%.1f max=%.1f std=%.6f",
        horizons,
        summary["mean_sindy_valid_time"],
        summary["min_sindy_valid_time"],
        summary["max_sindy_valid_time"],
        summary["std_sindy_valid_time"],
    )
    logger.info("wrote %s", output_dir)


if __name__ == "__main__":
    main()
