#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
FORECASTING_SRC = REPO_ROOT / "forecasting" / "src"

for _bootstrap_path in (REPO_ROOT, FORECASTING_SRC):
    if str(_bootstrap_path) not in sys.path:
        sys.path.insert(0, str(_bootstrap_path))

from common.logging_setup import add_logging_arguments, configure_from_args, get_logger, is_debug
from common.plotting import add_time_axis_argument
from forecasting.config import ConfigDict, load_config, require_key, require_section
from forecasting.visualization.runner import run_visualization_config


logger = get_logger("forecasting.scripts.plot_forecasting_visuals")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate config-driven reusable forecasting visualizations."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a JSON visualization config.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Override output.base_dir, e.g. to render previews outside the "
            "repository. Affects only where figures are written."
        ),
    )
    add_time_axis_argument(parser)
    add_logging_arguments(parser)
    return parser.parse_args()


def _get_logging_level_from_config(cfg: ConfigDict) -> str:
    logging_cfg = require_section(cfg, "logging")
    level = str(require_key(logging_cfg, "level", "logging")).upper()

    allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if level not in allowed:
        raise ValueError(f"logging.level must be one of {sorted(allowed)}, got {level!r}.")

    return level


def main() -> int:
    args = _parse_args()
    configure_from_args(args, component="PLOTS")

    try:
        cfg = load_config(args.config)
        configure_from_args(args, component="PLOTS", config_level=_get_logging_level_from_config(cfg))

        run_visualization_config(
            cfg,
            repo_root=REPO_ROOT,
            config_path=args.config,
            time_axis_mode=args.time_axis_mode,
            output_root=args.output_root,
        )

        return 0

    except Exception as exc:
        if is_debug():
            logger.exception("Forecasting visualization failed")
        else:
            logger.error("Forecasting visualization failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
