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
from ae_sindy.configs import AESINDyConfig
from ae_sindy.data_utils import load_array, prepare_kse_datasets, save_normalization_stats
from ae_sindy.logging_utils import get_logger
from ae_sindy.train import train_network_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AE-SINDy training from a config file")
    parser.add_argument("--config", required=True, help="Path to AE-SINDy experiment config JSON")
    add_logging_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AESINDyConfig.from_json(args.config)

    if not config.data_file:
        raise ValueError("config.data_file is required")

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    _settings = resolve_logging_settings(args, config_level=config.log_level)
    configure_logging(_settings.level, _settings.log_file or run_dir / "train.log", reset=True)
    logger = get_logger("run_train")

    config.save_json(run_dir / "config.json")
    config.save_pickle(run_dir / "params.pkl")

    logger.info("loading data from %s", config.data_file)
    series = load_array(config.data_file)
    logger.info(
        "Input trajectory: shape=%s, representation=full state, dtype=%s",
        tuple(series.shape), series.dtype,
    )

    train_data, val_data, _, normalization_stats = prepare_kse_datasets(
        series,
        dt=config.dt,
        model_order=config.model_order,
        train_slice=config.train_slice,
        val_slice=config.val_slice,
        test_slice=None,
        normalize=config.normalize,
    )
    if train_data is None or val_data is None:
        raise ValueError("config.train_slice and config.val_slice are required for training")

    if config.normalize and normalization_stats is not None:
        save_normalization_stats(normalization_stats, run_dir / "normalization_stats.json")
        logger.info(
            "saved train-fitted normalization stats mean=%.8e std=%.8e",
            normalization_stats.mean,
            normalization_stats.std,
        )

    results = train_network_torch(train_data, val_data, config, run_dir=run_dir)
    logger.info(
        "training complete run_dir=%s final_decoder=%.6e",
        run_dir,
        results["loss_decoder"],
    )


if __name__ == "__main__":
    main()
