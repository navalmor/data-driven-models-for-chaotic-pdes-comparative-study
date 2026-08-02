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
from forecasting.config import ConfigDict, load_config, require_key, require_section
from forecasting.runners.ngrc_runner import run_ngrc_experiment


logger = get_logger("forecasting.scripts.run_ngrc")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one config-driven NGRC forecasting experiment."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the final JSON experiment config.",
    )
    add_logging_arguments(parser)
    return parser.parse_args()


def _get_logging_level_from_config(cfg: ConfigDict) -> str:
    logging_cfg = require_section(cfg, "logging")
    level = str(require_key(logging_cfg, "level", "logging")).upper()

    allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if level not in allowed:
        raise ValueError(
            f"logging.level must be one of {sorted(allowed)}, got {level!r}."
        )

    return level


def _get_append_registry_from_config(cfg: ConfigDict) -> bool:
    """
    Return whether this single run should append to the final registry.

    Final selected runs should set:
        output.append_registry = true

    Smoke/probe/debug runs should set:
        output.append_registry = false

    Missing value defaults to true for backward compatibility with existing
    selected configs, but final configs should specify it explicitly.
    """
    output_cfg = require_section(cfg, "output")
    value = output_cfg.get("append_registry", True)

    if not isinstance(value, bool):
        raise TypeError(
            f"output.append_registry must be a boolean if provided, got {value!r}."
        )

    return value



def _get_evaluate_test_from_config(cfg: ConfigDict) -> bool:
    """
    Return whether this single run should evaluate the test split.

    Final selected and smoke runs should usually set:
        output.evaluate_test = true

    Probe/debug runs can set:
        output.evaluate_test = false

    This prevents exploratory runs from touching the final test set.
    """
    output_cfg = require_section(cfg, "output")
    value = output_cfg.get("evaluate_test", True)

    if not isinstance(value, bool):
        raise TypeError(
            f"output.evaluate_test must be a boolean if provided, got {value!r}."
        )

    return value


def main() -> int:
    args = _parse_args()

    # Configure logging up front (component NGRC) so a failure while loading the
    # config is still reported through the shared format; then re-resolve so the
    # config's logging.level applies when no CLI/env override was given.
    configure_from_args(args, component="NGRC")

    try:
        cfg = load_config(args.config)
        configure_from_args(args, component="NGRC", config_level=_get_logging_level_from_config(cfg))

        run_ngrc_experiment(
            cfg,
            repo_root=REPO_ROOT,
            config_path=args.config,
            append_registry=_get_append_registry_from_config(cfg),
            evaluate_test=_get_evaluate_test_from_config(cfg),
        )

        return 0

    except Exception as exc:
        # Concise by default; full traceback under --debug.
        if is_debug():
            logger.exception("NGRC run failed")
        else:
            logger.error("NGRC run failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
