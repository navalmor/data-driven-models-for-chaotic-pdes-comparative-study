from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _bootstrap_path in (_REPO_ROOT, _REPO_ROOT / "ae_sindy"):
    if str(_bootstrap_path) not in sys.path:
        sys.path.insert(0, str(_bootstrap_path))

from common.logging_setup import add_logging_arguments, resolve_logging_settings
from common.plotting import add_time_axis_argument
from ae_sindy import configure_logging
from ae_sindy.configs import AESINDyConfig
from ae_sindy.logging_utils import get_logger
from ae_sindy.visualization import visualize_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AE-SINDy figures from a config file")
    parser.add_argument("--config", required=True, help="Path to AE-SINDy experiment config JSON")
    add_time_axis_argument(parser)
    add_logging_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AESINDyConfig.from_json(args.config)

    run_dir = config.run_dir
    output_dir = (
        Path(config.visualization_output_dir)
        if config.visualization_output_dir is not None
        else run_dir / "figures"
    )

    _settings = resolve_logging_settings(args, config_level=config.log_level)
    configure_logging(_settings.level, _settings.log_file or run_dir / "visualize.log", reset=True)
    logger = get_logger("run_visualize")

    result = visualize_run(
        run_dir,
        data_file=config.data_file,
        output_dir=output_dir,
        show=config.visualization_show,
        config=config,
        time_axis_mode=args.time_axis_mode,
    )
    logger.info("visualization complete output_dir=%s", result["output_dir"])


if __name__ == "__main__":
    main()
