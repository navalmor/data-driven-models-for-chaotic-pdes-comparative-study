from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

ConfigDict = dict[str, Any]


def _json_default(value: Any) -> Any:
    """
    Convert non-standard lightweight Python objects into JSON-safe values.

    This is mainly used when saving resolved experiment configs, where paths
    and NumPy scalar values may appear after runtime resolution.
    """
    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable."
    )


def load_config(path: str | Path) -> ConfigDict:
    """
    Load a strict JSON configuration file.

    The final forecasting pipeline uses JSON configs only. TOML/YAML/INI files
    are intentionally rejected to keep the project configuration format
    consistent and reproducible.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    if path.suffix.lower() != ".json":
        raise ValueError(
            f"Unsupported config format: {path.suffix}. "
            "Final forecasting configs must use .json files."
        )

    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config file: {path}") from exc

    if not isinstance(cfg, dict):
        raise TypeError(f"Top-level config must be a JSON object: {path}")

    logger.debug(
        "Loaded config | path=%s sections=%s",
        path,
        sorted(cfg.keys()),
    )

    return cfg


def require_section(cfg: ConfigDict, name: str) -> ConfigDict:
    """
    Return a required top-level config section.

    Example:
        experiment = require_section(cfg, "experiment")
    """
    if name not in cfg:
        raise KeyError(f"Missing required config section: '{name}'")

    section = cfg[name]

    if not isinstance(section, dict):
        raise TypeError(f"Config section '{name}' must be a JSON object.")

    return section


def require_key(section: ConfigDict, key: str, section_name: str) -> Any:
    """
    Return a required key from a config section.

    Example:
        run_id = require_key(experiment, "run_id", "experiment")
    """
    if key not in section:
        raise KeyError(f"Missing required key '{key}' in section '{section_name}'")

    return section[key]


def validate_experiment(
    cfg: ConfigDict,
    *,
    expected_type: str,
    expected_model: str,
) -> None:
    """
    Validate common experiment identity fields.

    This prevents accidentally running, for example, an RC config through the
    NGRC runner or a search config through a single-run script.
    """
    experiment = require_section(cfg, "experiment")

    actual_type = require_key(experiment, "type", "experiment")
    actual_model = require_key(experiment, "model", "experiment")

    if actual_type != expected_type:
        raise ValueError(
            f"Expected experiment.type='{expected_type}', got '{actual_type}'"
        )

    if actual_model != expected_model:
        raise ValueError(
            f"Expected experiment.model='{expected_model}', got '{actual_model}'"
        )

    logger.debug(
        "Validated experiment identity | type=%s model=%s",
        actual_type,
        actual_model,
    )


def save_resolved_config_json(path: str | Path, cfg: ConfigDict) -> None:
    """
    Save the exact resolved config used by an experiment.

    Each experiment run should save a config snapshot so that the run can be
    reproduced later without guessing runtime-resolved paths or parameters.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            cfg,
            f,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        f.write("\n")

    logger.debug("Saved resolved config | path=%s", path)
