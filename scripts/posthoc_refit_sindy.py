from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _bootstrap_path in (_REPO_ROOT, _REPO_ROOT / "ae_sindy"):
    if str(_bootstrap_path) not in sys.path:
        sys.path.insert(0, str(_bootstrap_path))

import numpy as np
import torch

from common.logging_setup import add_logging_arguments, configure_from_args, get_logger
from ae_sindy.analysis import load_config_from_run, load_model_checkpoint, run_analysis, run_inference
from ae_sindy.configs import AESINDyConfig
from ae_sindy.data_utils import load_array, load_normalization_stats, prepare_kse_datasets
from ae_sindy.model import SINDyAutoencoderModel

logger = get_logger("posthoc_sindy")


MODELS = {
    "latent6_trial019": {
        "run_dir": "experiments/optuna_stage2/runs/optuna_stage2_trial_019",
    },
    "latent8_trial014": {
        "run_dir": "experiments/optuna_stage4_latent8/runs/optuna_stage4_latent8_trial_014",
    },
    "latent8_trial014_seed7": {
        "run_dir": "experiments/seed_robustness/stage4_latent8_trial014/runs/stage4_latent8_trial014_seed_7",
    },
    "latent8_trial014_seed123": {
        "run_dir": "experiments/seed_robustness/stage4_latent8_trial014/runs/stage4_latent8_trial014_seed_123",
    },
    "latent8_trial014_seed2025": {
        "run_dir": "experiments/seed_robustness/stage4_latent8_trial014/runs/stage4_latent8_trial014_seed_2025",
    },
    "latent6_trial012": {
        "run_dir": "experiments/optuna_stage2/runs/optuna_stage2_trial_012",
    },
}


# H-01 safeguard: threshold/ridge SELECTION must use VALIDATION windows. The test
# windows [25000,30000] are for the explicit final-test/report mode only and must
# never be used for selection by accident.
VALIDATION_WINDOWS: List[Tuple[int, int]] = [
    (20000, 21000),
    (21000, 22000),
    (22000, 23000),
    (23000, 24000),
    (24000, 25000),
]
TEST_WINDOWS: List[Tuple[int, int]] = [
    (25000, 26000),
    (26000, 27000),
    (27000, 28000),
    (28000, 29000),
    (29000, 30000),
]
# Default windows for threshold/ridge SELECTION are the validation windows.
WINDOWS: List[Tuple[int, int]] = VALIDATION_WINDOWS


def load_refit_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a JSON object: {path}")

    return data


def resolve_model_specs(model_items: object, cli_models: List[str]) -> List[Dict[str, str]]:
    specs: List[Dict[str, str]] = []

    if model_items is None:
        for model_name in cli_models:
            if model_name not in MODELS:
                raise ValueError(f"Unknown model {model_name}. Available: {list(MODELS)}")
            specs.append({
                "name": model_name,
                "run_dir": str(MODELS[model_name]["run_dir"]),
            })
        return specs

    if not isinstance(model_items, list):
        raise ValueError("Config field 'models' must be a list")

    for item in model_items:
        if isinstance(item, str):
            if item not in MODELS:
                raise ValueError(f"Unknown model {item}. Available: {list(MODELS)}")
            specs.append({
                "name": item,
                "run_dir": str(MODELS[item]["run_dir"]),
            })
            continue

        if not isinstance(item, dict):
            raise ValueError("Each config model must be either a string or an object")

        name = str(item["name"])
        run_dir = item.get("run_dir")

        if run_dir is None:
            if name not in MODELS:
                raise ValueError(f"Model {name} has no run_dir and is not available in MODELS")
            run_dir = MODELS[name]["run_dir"]

        specs.append({
            "name": name,
            "run_dir": str(run_dir),
        })

    if not specs:
        raise ValueError("No models requested")

    return specs


def resolve_windows(
    window_items: object,
    default_windows: List[Tuple[int, int]] | None = None,
) -> List[Tuple[int, int]]:
    if window_items is None:
        return list(default_windows if default_windows is not None else WINDOWS)

    if not isinstance(window_items, list):
        raise ValueError("Config field 'windows' must be a list")

    windows: List[Tuple[int, int]] = []
    for item in window_items:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("Each window must be a two-element list [start, end]")
        start = int(item[0])
        end = int(item[1])
        if end <= start:
            raise ValueError(f"Invalid window [{start}, {end}]")
        windows.append((start, end))

    if not windows:
        raise ValueError("No windows requested")

    return windows


def write_config_used(
    output_root: Path,
    config_path: Path | None,
    model_specs: List[Dict[str, str]],
    thresholds: List[float],
    ridge_alpha: float,
    max_iter: int,
    windows: List[Tuple[int, int]],
    windows_preset: str = "validation",
) -> None:
    # H-01 provenance: mark whether these windows are the validation SELECTION set
    # or the explicit test REPORTING set (derived from the windows themselves so it
    # is correct even when windows come from an explicit config override).
    is_test = any(int(start) >= 25000 for start, _ in windows)
    config_used = {
        "config_path": str(config_path) if config_path is not None else None,
        "models": model_specs,
        "thresholds": thresholds,
        "ridge_alpha": ridge_alpha,
        "max_iter": max_iter,
        "windows": [[start, end] for start, end in windows],
        "windows_preset": windows_preset,
        "windows_role": "test_report" if is_test else "validation_selection",
        "test_used_for_selection": False,
        "selection_note": (
            "Threshold/ridge selection uses validation windows [20000,25000]; "
            "test windows [25000,30000] are for final reporting only (H-01 safeguard)."
        ),
        "output_root": str(output_root),
    }

    with (output_root / "posthoc_sindy_refit_config_used.json").open("w", encoding="utf-8") as f:
        json.dump(config_used, f, indent=2)
        f.write("\n")


def ridge_solve(theta: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Solve min ||theta xi - y||^2 + alpha ||xi||^2."""
    n_features = theta.shape[1]
    lhs = theta.T @ theta + alpha * np.eye(n_features, dtype=theta.dtype)
    rhs = theta.T @ y
    return np.linalg.solve(lhs, rhs)


def stlsq(
    theta: np.ndarray,
    dz: np.ndarray,
    threshold: float,
    alpha: float = 1e-6,
    max_iter: int = 10,
) -> np.ndarray:
    """Sequential thresholded least squares for dz ≈ theta @ Xi."""
    theta = np.asarray(theta, dtype=np.float64)
    dz = np.asarray(dz, dtype=np.float64)

    xi = ridge_solve(theta, dz, alpha=alpha)

    for _ in range(max_iter):
        small = np.abs(xi) < threshold
        xi[small] = 0.0

        for dim in range(dz.shape[1]):
            support = np.abs(xi[:, dim]) >= threshold
            if not np.any(support):
                xi[:, dim] = 0.0
                continue

            xi_support = ridge_solve(theta[:, support], dz[:, dim : dim + 1], alpha=alpha).ravel()
            xi[:, dim] = 0.0
            xi[support, dim] = xi_support

    xi[np.abs(xi) < threshold] = 0.0
    return xi.astype(np.float32)


def load_training_latent_system(src_run_dir: Path) -> tuple[AESINDyConfig, Dict[str, object], OrderedDict, np.ndarray, np.ndarray, np.ndarray]:
    cfg = load_config_from_run(src_run_dir)
    params = cfg.to_params_dict(run_dir=src_run_dir)
    params["coefficient_mask"] = np.asarray(params["coefficient_mask"], dtype=np.float32)

    model = SINDyAutoencoderModel(params)
    checkpoint_path = load_model_checkpoint(model, src_run_dir, cfg.run_name)
    state = torch.load(checkpoint_path, map_location="cpu")

    normalization_stats = None
    if cfg.normalize:
        normalization_stats = load_normalization_stats(src_run_dir / "normalization_stats.json")

    series = load_array(cfg.data_file)

    train_data, _, _, _ = prepare_kse_datasets(
        series,
        dt=cfg.dt,
        model_order=cfg.model_order,
        train_slice=tuple(cfg.train_slice),
        val_slice=None,
        test_slice=None,
        normalize=cfg.normalize,
        normalization_stats=normalization_stats,
    )

    if train_data is None:
        raise ValueError(f"No train_data produced for {src_run_dir}")

    results = run_inference(model, train_data, params, device=torch.device("cpu"))

    theta = np.asarray(results["Theta"], dtype=np.float64)
    dz = np.asarray(results["dz"], dtype=np.float64)
    z = np.asarray(results["z"], dtype=np.float64)

    return cfg, params, state, theta, dz, z


def write_refit_run(
    cfg: AESINDyConfig,
    params: Dict[str, object],
    original_state: OrderedDict,
    xi_refit: np.ndarray,
    dst_run_dir: Path,
    src_run_dir: Path,
    threshold: float,
) -> None:
    dst_run_dir.mkdir(parents=True, exist_ok=True)

    new_state = OrderedDict()
    for key, value in original_state.items():
        if isinstance(value, torch.Tensor):
            new_state[key] = value.detach().cpu().clone()
        else:
            new_state[key] = value

    new_state["sindy_coefficients"] = torch.as_tensor(xi_refit, dtype=torch.float32)
    torch.save(new_state, dst_run_dir / "model.pt")

    if cfg.normalize:
        shutil.copy2(src_run_dir / "normalization_stats.json", dst_run_dir / "normalization_stats.json")

    mask = (np.abs(xi_refit) > 0).astype(np.float32)

    params_out = dict(params)
    params_out["coefficient_mask"] = mask
    params_out["coefficient_threshold"] = float(threshold)
    params_out["run_name"] = dst_run_dir.name
    params_out["save_name"] = dst_run_dir.name
    params_out["output_dir"] = str(dst_run_dir.parent)
    params_out["data_path"] = str(dst_run_dir) + "/"

    with (dst_run_dir / "params.pkl").open("wb") as f:
        pickle.dump(params_out, f)

    # Do not write config.json here because params_out may contain runtime objects
    # such as torch.device that are not JSON serializable. run_analysis() loads
    # params.pkl first, so params.pkl is the authoritative config for refit runs.


def evaluate_refit_window(
    cfg: AESINDyConfig,
    src_run_dir: Path,
    refit_window_dir: Path,
    window_index: int,
    window: Tuple[int, int],
) -> Dict[str, object]:
    normalization_stats = None
    if cfg.normalize:
        normalization_stats = load_normalization_stats(refit_window_dir / "normalization_stats.json")

    series = load_array(cfg.data_file)

    _, _, test_data, _ = prepare_kse_datasets(
        series,
        dt=cfg.dt,
        model_order=cfg.model_order,
        train_slice=None,
        val_slice=None,
        test_slice=window,
        normalize=cfg.normalize,
        normalization_stats=normalization_stats,
    )

    if test_data is None:
        raise ValueError(f"Window {window} did not produce test_data")

    summary = run_analysis(
        refit_window_dir,
        test_data,
        dt=cfg.dt,
        device="cpu",
        generate_plots=False,
        rollout_max_steps=cfg.analysis_rollout_max_steps,
    )

    meta = summary.get("rollout_meta", {})

    return {
        "window_index": window_index,
        "window_start": window[0],
        "window_end": window[1],
        "sindy_valid_time": summary.get("sindy_valid_time"),
        "sindy_first_invalid_time": summary.get("sindy_first_invalid_time"),
        "encoder_valid_time": summary.get("encoder_valid_time"),
        "encoder_rollout_valid_time": summary.get("encoder_rollout_valid_time"),
        "decoder_error": summary.get("decoder_error"),
        "active_terms": summary.get("active_terms"),
        "sparsity_percent": summary.get("sparsity_percent"),
        "solver_success": meta.get("solver_success"),
        "rollout_success": meta.get("rollout_success"),
        "used_fallback": meta.get("used_fallback"),
        "used_steps": meta.get("used_steps"),
        "rollout_status": meta.get("rollout_status"),
        "run_dir": str(refit_window_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc sparse SINDy refit for trained AE-SINDy models")
    parser.add_argument("--models", nargs="*", default=["latent6_trial019", "latent8_trial014"])
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.03, 0.05, 0.07, 0.09, 0.11, 0.13])
    parser.add_argument("--ridge-alpha", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--output-root", default="experiments/posthoc_sindy_refit")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--windows-preset",
        choices=["validation", "test"],
        default="validation",
        help=(
            "Default window preset used ONLY when the config does not provide explicit "
            "'windows'. 'validation' (default) = five windows in [20000,25000] for "
            "threshold/ridge SELECTION. 'test' = five windows in [25000,30000] for the "
            "explicit final-test report AFTER the threshold/ridge are locked. "
            "H-01 safeguard: never select thresholds on the test window."
        ),
    )
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_from_args(args, component="SINDy")

    config_data: Dict[str, Any] = {}
    if args.config is not None:
        config_data = load_refit_config(args.config)

    output_root = Path(config_data.get("output_root", args.output_root))
    thresholds = [float(v) for v in config_data.get("thresholds", args.thresholds)]
    ridge_alpha = float(config_data.get("ridge_alpha", args.ridge_alpha))
    max_iter = int(config_data.get("max_iter", args.max_iter))
    default_windows = TEST_WINDOWS if args.windows_preset == "test" else VALIDATION_WINDOWS
    windows = resolve_windows(config_data.get("windows"), default_windows=default_windows)
    model_specs = resolve_model_specs(config_data.get("models"), list(args.models))

    output_root.mkdir(parents=True, exist_ok=True)
    write_config_used(
        output_root=output_root,
        config_path=args.config,
        model_specs=model_specs,
        thresholds=thresholds,
        ridge_alpha=ridge_alpha,
        max_iter=max_iter,
        windows=windows,
        windows_preset=args.windows_preset,
    )

    rows: List[Dict[str, object]] = []

    for model_spec in model_specs:
        model_name = model_spec["name"]
        src_run_dir = Path(model_spec["run_dir"])

        logger.info("Loading latent system for %s", model_name)
        logger.info("Source run: %s", src_run_dir)
        cfg, params, original_state, theta, dz, z = load_training_latent_system(src_run_dir)

        logger.info(
            "Latent trajectory (training window): shape=%s, dtype=%s, "
            "latent_dim=%s, poly_order=%s",
            tuple(z.shape), z.dtype, cfg.latent_dim, cfg.poly_order,
        )
        logger.debug(
            "Library matrix: theta=%s, derivative targets dz=%s",
            tuple(theta.shape), tuple(dz.shape),
        )

        for threshold in thresholds:
            logger.info("%s threshold=%.4f", model_name, threshold)

            xi_refit = stlsq(theta, dz, threshold=threshold, alpha=ridge_alpha, max_iter=max_iter)
            train_mse = float(np.mean((theta @ xi_refit - dz) ** 2))
            active_terms_refit = int(np.sum(np.abs(xi_refit) > 0))
            sparsity_refit = 100.0 * (1.0 - active_terms_refit / xi_refit.size)

            logger.info(
                "METRIC | Active coefficients: %d | sparsity=%.2f%% train_mse=%.6e",
                active_terms_refit, sparsity_refit, train_mse,
            )

            threshold_tag = f"thr_{threshold:.3f}".replace(".", "p")
            refit_root = output_root / model_name / threshold_tag

            for window_index, window in enumerate(windows):
                refit_window_dir = refit_root / f"window_{window_index:02d}_{window[0]}_{window[1]}"

                write_refit_run(
                    cfg=cfg,
                    params=params,
                    original_state=original_state,
                    xi_refit=xi_refit,
                    dst_run_dir=refit_window_dir,
                    src_run_dir=src_run_dir,
                    threshold=threshold,
                )

                row = evaluate_refit_window(
                    cfg=cfg,
                    src_run_dir=src_run_dir,
                    refit_window_dir=refit_window_dir,
                    window_index=window_index,
                    window=window,
                )

                row.update({
                    "model": model_name,
                    "threshold": threshold,
                    "ridge_alpha": ridge_alpha,
                    "refit_train_mse": train_mse,
                    "refit_active_terms": active_terms_refit,
                    "refit_sparsity_percent": sparsity_refit,
                })
                rows.append(row)

                logger.info(
                    "%s threshold=%.3f window=%d sindy_t=%s active=%s solver_success=%s",
                    model_name, threshold, window_index,
                    row["sindy_valid_time"], row["active_terms"], row["solver_success"],
                )

    detail_path = output_root / "posthoc_sindy_refit_detail.csv"
    with detail_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    keys = sorted({(r["model"], r["threshold"]) for r in rows})
    for model_name, threshold in keys:
        subset = [r for r in rows if r["model"] == model_name and r["threshold"] == threshold]
        sindy = np.array([float(r["sindy_valid_time"] or 0.0) for r in subset], dtype=float)
        decoder = np.array([float(r["decoder_error"] or np.nan) for r in subset], dtype=float)
        active = np.array([float(r["active_terms"] or np.nan) for r in subset], dtype=float)
        solver_success_count = sum(1 for r in subset if bool(r["solver_success"]))

        summary_rows.append({
            "model": model_name,
            "threshold": threshold,
            "n_windows": len(subset),
            "mean_sindy_valid_time": float(np.mean(sindy)),
            "min_sindy_valid_time": float(np.min(sindy)),
            "max_sindy_valid_time": float(np.max(sindy)),
            "std_sindy_valid_time": float(np.std(sindy)),
            "mean_decoder_error": float(np.nanmean(decoder)),
            "mean_active_terms": float(np.nanmean(active)),
            "solver_success_count": solver_success_count,
            "refit_train_mse": float(subset[0]["refit_train_mse"]),
            "refit_active_terms": int(subset[0]["refit_active_terms"]),
            "refit_sparsity_percent": float(subset[0]["refit_sparsity_percent"]),
        })

    summary_path = output_root / "posthoc_sindy_refit_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    logger.info("Saved detail:  %s", detail_path)
    logger.info("Saved summary: %s", summary_path)


if __name__ == "__main__":
    main()
