#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
FORECASTING_SRC = REPO_ROOT / "forecasting" / "src"

for _bootstrap_path in (REPO_ROOT, FORECASTING_SRC):
    if str(_bootstrap_path) not in sys.path:
        sys.path.insert(0, str(_bootstrap_path))

from common.logging_setup import add_logging_arguments, configure_from_args, get_logger, is_debug
from forecasting.runners.pinn_runner import run_pinn_experiment


logger = get_logger("forecasting.scripts.run_pinn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single PINN forecasting experiment.")
    parser.add_argument("--config", required=True, help="Path to PINN config JSON.")
    add_logging_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_from_args(args, component="PINN")

    try:
        repo_root = Path.cwd()
        config_path = Path(args.config)
        cfg = json.loads(config_path.read_text())

        # The PINN runner (component PINN) emits the full lifecycle: input shape,
        # device, training progress, primary metrics, output path and elapsed time.
        output_cfg = cfg.get("output", {})
        result = run_pinn_experiment(
            cfg,
            repo_root=repo_root,
            config_path=config_path,
            append_registry=bool(output_cfg.get("append_registry", False)),
            evaluate_test=bool(output_cfg.get("evaluate_test", False)),
        )
        logger.debug("summary written to %s", result.summary_path)

        return 0

    except Exception as exc:
        if is_debug():
            logger.exception("PINN run failed")
        else:
            logger.error("PINN run failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
