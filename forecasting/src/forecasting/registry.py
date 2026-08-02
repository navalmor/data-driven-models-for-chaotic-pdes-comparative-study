from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


REGISTRY_COLUMNS = [
    "run_id",
    "model_name",
    "run_type",
    "data_mode",
    "native_dimension",
    "physical_dimension",
    "train_end_index",
    "valid_end_index",
    "standardized",
    "decoded_to_physical",
    "selection_metric",
    "valid_relative_l2_horizon_time",
    "test_relative_l2_horizon_time",
    "valid_relative_l2_mean",
    "test_relative_l2_mean",
    "valid_relative_l2_final",
    "test_relative_l2_final",
    "config_path",
    "output_dir",
]


REQUIRED_REGISTRY_COLUMNS = [
    "run_id",
    "model_name",
    "run_type",
    "data_mode",
    "native_dimension",
    "physical_dimension",
    "selection_metric",
    "output_dir",
]


def _format_registry_value(value: Any) -> str:
    """
    Convert Python values into stable CSV-friendly strings.

    The registry is intentionally compact. Missing or non-finite values are
    written as empty cells instead of strings such as 'nan' or 'inf'.
    """
    if value is None:
        return ""

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)

    if isinstance(value, float):
        return repr(value) if math.isfinite(value) else ""

    # NumPy scalar support without importing NumPy in this lightweight module.
    if hasattr(value, "item"):
        try:
            return _format_registry_value(value.item())
        except Exception:
            pass

    if isinstance(value, Path):
        return str(value)

    return str(value)


def validate_registry_row(row: dict[str, Any]) -> None:
    """
    Validate one registry row before writing.

    Missing optional columns are allowed and will be written as empty values.
    Unknown columns are not allowed because they usually indicate outdated
    pipeline keys or inconsistent metric naming.
    """
    allowed = set(REGISTRY_COLUMNS)
    received = set(row)

    unknown = sorted(received - allowed)
    if unknown:
        raise ValueError(
            "Registry row contains unknown column(s): "
            + ", ".join(unknown)
            + ". Update the row construction or REGISTRY_COLUMNS intentionally."
        )

    missing_required = [
        column
        for column in REQUIRED_REGISTRY_COLUMNS
        if row.get(column) in (None, "")
    ]

    if missing_required:
        raise ValueError(
            "Registry row is missing required column(s): "
            + ", ".join(missing_required)
        )


def normalize_registry_row(row: dict[str, Any]) -> dict[str, str]:
    """
    Return a complete registry row in the fixed final column order.
    """
    validate_registry_row(row)

    return {
        column: _format_registry_value(row.get(column))
        for column in REGISTRY_COLUMNS
    }


def _validate_existing_header(path: Path) -> None:
    """
    Ensure an existing registry file uses the final expected schema.

    This prevents silently appending final pipeline rows to an older registry
    format that still contains legacy metrics or outdated column names.
    """
    if not path.exists() or path.stat().st_size == 0:
        return

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        existing_header = next(reader, None)

    if existing_header != REGISTRY_COLUMNS:
        raise ValueError(
            f"Existing registry header does not match final schema: {path}\n"
            f"Expected: {REGISTRY_COLUMNS}\n"
            f"Found:    {existing_header}\n"
            "Archive/delete the old registry or intentionally migrate it before appending."
        )


def append_registry_row(path: str | Path, row: dict[str, Any]) -> None:
    """
    Append one experiment result to the global forecasting registry.

    The registry is a compact, model-independent CSV table for thesis-level
    comparison across NGRC, RC, PINN, and PERC.

    Detailed model-specific hyperparameters should stay in:
        - config_resolved.json
        - summary.json
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _validate_existing_header(path)

    normalized_row = normalize_registry_row(row)
    file_exists = path.exists() and path.stat().st_size > 0

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)

        if not file_exists:
            writer.writeheader()

        writer.writerow(normalized_row)

    logger.debug(
        "Appended registry row | path=%s run_id=%s model=%s mode=%s",
        path,
        normalized_row["run_id"],
        normalized_row["model_name"],
        normalized_row["data_mode"],
    )
