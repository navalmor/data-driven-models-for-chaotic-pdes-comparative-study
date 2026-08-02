from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _bootstrap_path in (_REPO_ROOT, _REPO_ROOT / "ae_sindy"):
    if str(_bootstrap_path) not in sys.path:
        sys.path.insert(0, str(_bootstrap_path))

from common.logging_setup import add_logging_arguments, resolve_logging_settings
from ae_sindy import configure_logging
from ae_sindy.analysis import run_analysis
from ae_sindy.configs import AESINDyConfig
from ae_sindy.data_utils import load_array, load_normalization_stats, prepare_kse_datasets
from ae_sindy.logging_utils import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AE-SINDy analysis from a config file")
    parser.add_argument("--config", required=True, help="Path to AE-SINDy experiment config JSON")
    parser.add_argument(
        "--eval-split",
        choices=["val", "test"],
        default="test",
        help=(
            "Which data split to analyze. 'val' uses config.val_slice and writes "
            "analysis_summary_val.pkl (use this for model/hyperparameter SELECTION). "
            "'test' uses config.test_slice and writes analysis_summary.pkl + "
            "analysis_summary_test.pkl (one-shot held-out REPORTING). "
            "H-01 safeguard: selection must use 'val'; the test split must never be "
            "used for selection. Default is 'test' for backward compatibility."
        ),
    )
    add_logging_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AESINDyConfig.from_json(args.config)

    if not config.data_file:
        raise ValueError("config.data_file is required")

    # H-01 safeguard: pick the evaluation window from the requested split. A
    # validation analysis (selection) must never fall back to the test window.
    eval_split = args.eval_split
    if eval_split == "val":
        if config.val_slice is None:
            raise ValueError("config.val_slice is required for --eval-split val")
        eval_slice = config.val_slice
    else:
        if config.test_slice is None:
            raise ValueError("config.test_slice is required for --eval-split test")
        eval_slice = config.test_slice

    run_dir = config.run_dir
    _settings = resolve_logging_settings(args, config_level=config.log_level)
    configure_logging(_settings.level, _settings.log_file or run_dir / "analysis.log", reset=True)
    logger = get_logger("run_analysis")

    normalization_stats = None
    if config.normalize:
        stats_path = run_dir / "normalization_stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Missing {stats_path}. Re-train this run with train-fitted normalization before analysis."
            )
        normalization_stats = load_normalization_stats(stats_path)
        logger.info(
            "loaded train-fitted normalization stats mean=%.8e std=%.8e",
            normalization_stats.mean,
            normalization_stats.std,
        )

    logger.info("loading data from %s eval_split=%s slice=%s", config.data_file, eval_split, eval_slice)
    series = load_array(config.data_file)
    logger.info(
        "Input trajectory: shape=%s, representation=full state, dtype=%s",
        tuple(series.shape), series.dtype,
    )

    _, _, eval_data, _ = prepare_kse_datasets(
        series,
        dt=config.dt,
        model_order=config.model_order,
        train_slice=None,
        val_slice=None,
        test_slice=eval_slice,
        normalize=config.normalize,
        normalization_stats=normalization_stats,
    )
    if eval_data is None:
        raise ValueError(f"eval_split={eval_split} did not produce data for slice {eval_slice}")

    summary = run_analysis(
        run_dir,
        eval_data,
        dt=config.dt,
        device=config.device,
        generate_plots=config.analysis_generate_plots,
        rollout_max_steps=config.analysis_rollout_max_steps,
        eval_split=eval_split,
    )
    logger.info(
        "analysis complete eval_split=%s checkpoint=%s decoder_error=%.6e quality=%s main_quality_metric=%s",
        eval_split,
        summary["checkpoint_path"],
        summary["decoder_error"],
        summary.get("run_quality"),
        summary.get("main_quality_metric"),
    )


if __name__ == "__main__":
    main()
