from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forecasting.config import ConfigDict, require_key, require_section


logger = logging.getLogger(__name__)

Array = np.ndarray

SUPPORTED_DATA_MODES = {"full64", "latent8", "latent6"}
EXPECTED_PHYSICAL_DIMENSION = 64


@dataclass(frozen=True)
class ForecastData:
    """
    Forecasting data container.

    native_*:
        Space where the forecasting model is trained and rolled out.
        Examples: full64 KSE state, latent8, latent6.

    physical_*:
        Original physical KSE space used for final thesis evaluation.
        For this project, physical space is the 64D KSE state.
    """

    data_mode: str

    native_train: Array
    native_valid: Array
    native_test: Array

    physical_train: Array
    physical_valid: Array
    physical_test: Array

    native_dimension: int
    physical_dimension: int

    train_end_index: int
    valid_end_index: int

    physical_data_path: Path
    native_data_dir: Path | None
    decoder_run_dir: Path | None
    native_data_source: str


def _resolve_path(repo_root: Path, path: str | Path) -> Path:
    """
    Resolve path relative to repo_root unless it is already absolute.
    """
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _as_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got bool.")

    try:
        value_int = int(value)
    except Exception as exc:
        raise TypeError(f"{name} must be an integer, got {value!r}.") from exc

    if value_int <= 0:
        raise ValueError(f"{name} must be positive, got {value_int}.")

    return value_int


def _load_2d_finite_array(path: Path, *, name: str) -> Array:
    """
    Load a finite 2D NumPy array from disk.
    """
    if not path.exists():
        raise FileNotFoundError(f"{name} file does not exist: {path}")

    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr, dtype=np.float64)

    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D array with shape (time_steps, dimension), "
            f"got shape {arr.shape} at {path}."
        )

    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one timestep: {path}")

    if arr.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one state dimension: {path}")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf: {path}")

    logger.debug(
        "Loaded array | name=%s path=%s shape=%s",
        name,
        path,
        tuple(arr.shape),
    )

    return arr


def complete_trajectory_shape(train: Array, valid: Array, test: Array) -> tuple[int, int]:
    """Shape of the complete loaded trajectory, recovered from its splits.

    ``split_time_series`` partitions a series into ``series[:a]``, ``series[a:b]``
    and ``series[b:]``, so the three splits together are exactly the original
    trajectory. This reports that shape for logging without reloading or
    concatenating any data: only the lengths of arrays already in memory are read.
    """
    total_steps = int(len(train)) + int(len(valid)) + int(len(test))
    return total_steps, int(np.asarray(train).shape[1])


def split_time_series(
    series: Array,
    *,
    train_end_index: int,
    valid_end_index: int,
    name: str,
) -> tuple[Array, Array, Array]:
    """
    Split a complete time series into train/validation/test parts.

    Split convention:
        train = series[:train_end_index]
        valid = series[train_end_index:valid_end_index]
        test  = series[valid_end_index:]

    This chronological split avoids future-to-past leakage.
    """
    series = np.asarray(series, dtype=np.float64)

    if series.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D array with shape (time_steps, dimension), "
            f"got shape {series.shape}."
        )

    if not np.all(np.isfinite(series)):
        raise ValueError(f"{name} contains NaN or Inf.")

    train_end_index = _as_positive_int(train_end_index, name="train_end_index")
    valid_end_index = _as_positive_int(valid_end_index, name="valid_end_index")

    n_steps = int(series.shape[0])

    if not (0 < train_end_index < valid_end_index < n_steps):
        raise ValueError(
            f"Invalid split indices for {name}: "
            f"train_end_index={train_end_index}, "
            f"valid_end_index={valid_end_index}, "
            f"n_steps={n_steps}."
        )

    train = series[:train_end_index]
    valid = series[train_end_index:valid_end_index]
    test = series[valid_end_index:]

    logger.debug(
        "Split time series | name=%s train=%s valid=%s test=%s",
        name,
        tuple(train.shape),
        tuple(valid.shape),
        tuple(test.shape),
    )

    return train, valid, test


def _validate_same_feature_dimension(
    arrays: list[Array],
    *,
    names: list[str],
) -> int:
    dimensions = [int(arr.shape[1]) for arr in arrays]

    if len(set(dimensions)) != 1:
        joined = ", ".join(
            f"{name}={dimension}"
            for name, dimension in zip(names, dimensions)
        )
        raise ValueError(f"Inconsistent feature dimensions: {joined}.")

    return dimensions[0]


def _validate_matching_split_lengths(
    *,
    native_train: Array,
    native_valid: Array,
    native_test: Array,
    physical_train: Array,
    physical_valid: Array,
    physical_test: Array,
) -> None:
    expected = (
        int(physical_train.shape[0]),
        int(physical_valid.shape[0]),
        int(physical_test.shape[0]),
    )
    actual = (
        int(native_train.shape[0]),
        int(native_valid.shape[0]),
        int(native_test.shape[0]),
    )

    if actual != expected:
        raise ValueError(
            "Native and physical split lengths do not match. "
            f"native={actual}, physical={expected}."
        )


def _load_latent_splits_from_series(
    *,
    z_series_path: Path,
    train_end_index: int,
    valid_end_index: int,
    expected_total_steps: int,
) -> tuple[Array, Array, Array]:
    z_series = _load_2d_finite_array(z_series_path, name="latent z_series")

    if z_series.shape[0] != expected_total_steps:
        raise ValueError(
            f"Latent z_series length mismatch: expected {expected_total_steps}, "
            f"got {z_series.shape[0]} at {z_series_path}."
        )

    return split_time_series(
        z_series,
        train_end_index=train_end_index,
        valid_end_index=valid_end_index,
        name="latent z_series",
    )


def _load_latent_splits_from_files(
    *,
    native_data_dir: Path,
    expected_train_steps: int,
    expected_valid_steps: int,
    expected_test_steps: int,
) -> tuple[Array, Array, Array]:
    z_train = _load_2d_finite_array(
        native_data_dir / "z_train.npy",
        name="latent z_train",
    )
    z_valid = _load_2d_finite_array(
        native_data_dir / "z_valid.npy",
        name="latent z_valid",
    )
    z_test = _load_2d_finite_array(
        native_data_dir / "z_test.npy",
        name="latent z_test",
    )

    actual_lengths = (
        int(z_train.shape[0]),
        int(z_valid.shape[0]),
        int(z_test.shape[0]),
    )
    expected_lengths = (
        int(expected_train_steps),
        int(expected_valid_steps),
        int(expected_test_steps),
    )

    if actual_lengths != expected_lengths:
        raise ValueError(
            "Latent split lengths do not match expected physical split lengths. "
            f"actual={actual_lengths}, expected={expected_lengths}."
        )

    _validate_same_feature_dimension(
        [z_train, z_valid, z_test],
        names=["z_train", "z_valid", "z_test"],
    )

    return z_train, z_valid, z_test


def _load_latent_data(
    *,
    native_data_dir: Path,
    train_end_index: int,
    valid_end_index: int,
    expected_total_steps: int,
    expected_train_steps: int,
    expected_valid_steps: int,
    expected_test_steps: int,
) -> tuple[Array, Array, Array, str]:
    """
    Load latent data.

    Priority:
        1. z_series.npy
        2. z_train.npy, z_valid.npy, z_test.npy

    z_series.npy is preferred because it preserves the full chronological
    trajectory in one file and guarantees the same split convention as the
    physical 64D data.
    """
    if not native_data_dir.exists():
        raise FileNotFoundError(
            f"Native latent data directory does not exist: {native_data_dir}"
        )

    z_series_path = native_data_dir / "z_series.npy"

    if z_series_path.exists():
        z_train, z_valid, z_test = _load_latent_splits_from_series(
            z_series_path=z_series_path,
            train_end_index=train_end_index,
            valid_end_index=valid_end_index,
            expected_total_steps=expected_total_steps,
        )

        logger.debug(
            "Loaded latent data from z_series.npy | dir=%s",
            native_data_dir,
        )

        return z_train, z_valid, z_test, "z_series.npy"

    z_train, z_valid, z_test = _load_latent_splits_from_files(
        native_data_dir=native_data_dir,
        expected_train_steps=expected_train_steps,
        expected_valid_steps=expected_valid_steps,
        expected_test_steps=expected_test_steps,
    )

    logger.debug(
        "Loaded latent data from split files | dir=%s",
        native_data_dir,
    )

    return z_train, z_valid, z_test, "z_train.npy/z_valid.npy/z_test.npy"


def _expected_native_dimension_for_mode(data_mode: str) -> int:
    if data_mode == "full64":
        return 64

    if data_mode == "latent8":
        return 8

    if data_mode == "latent6":
        return 6

    raise ValueError(
        f"Unsupported data_mode='{data_mode}'. "
        f"Supported values are: {sorted(SUPPORTED_DATA_MODES)}."
    )


def _validate_data_mode(data_mode: str) -> str:
    data_mode = str(data_mode)

    if data_mode not in SUPPORTED_DATA_MODES:
        raise ValueError(
            f"Unsupported experiment.data_mode='{data_mode}'. "
            f"Supported values are: {sorted(SUPPORTED_DATA_MODES)}."
        )

    return data_mode


def load_forecast_data(cfg: ConfigDict, *, repo_root: str | Path) -> ForecastData:
    """
    Load train/validation/test data for forecasting experiments.

    Final evaluation always uses physical 64D KSE data.

    Forecasting native space depends on data_mode:
        - full64  : native space = physical 64D KSE space
        - latent8 : native space = AE-SINDy latent8 space
        - latent6 : native space = AE-SINDy latent6 space
    """
    repo_root = Path(repo_root).expanduser().resolve()

    experiment_cfg = require_section(cfg, "experiment")
    data_cfg = require_section(cfg, "data")

    data_mode = _validate_data_mode(
        require_key(experiment_cfg, "data_mode", "experiment")
    )

    train_end_index = _as_positive_int(
        require_key(data_cfg, "train_end_index", "data"),
        name="data.train_end_index",
    )
    valid_end_index = _as_positive_int(
        require_key(data_cfg, "valid_end_index", "data"),
        name="data.valid_end_index",
    )

    physical_data_path = _resolve_path(
        repo_root,
        require_key(data_cfg, "physical_data_path", "data"),
    )

    physical_series = _load_2d_finite_array(
        physical_data_path,
        name="physical KSE series",
    )

    physical_dimension = int(physical_series.shape[1])

    if physical_dimension != EXPECTED_PHYSICAL_DIMENSION:
        raise ValueError(
            f"Physical data dimension mismatch: expected "
            f"{EXPECTED_PHYSICAL_DIMENSION}, got {physical_dimension} at "
            f"{physical_data_path}."
        )

    physical_train, physical_valid, physical_test = split_time_series(
        physical_series,
        train_end_index=train_end_index,
        valid_end_index=valid_end_index,
        name="physical KSE series",
    )

    expected_native_dimension = _expected_native_dimension_for_mode(data_mode)

    if data_mode == "full64":
        native_dimension = physical_dimension

        if native_dimension != expected_native_dimension:
            raise ValueError(
                f"Native dimension mismatch for full64: expected "
                f"{expected_native_dimension}, got {native_dimension}."
            )

        result = ForecastData(
            data_mode=data_mode,
            native_train=physical_train,
            native_valid=physical_valid,
            native_test=physical_test,
            physical_train=physical_train,
            physical_valid=physical_valid,
            physical_test=physical_test,
            native_dimension=native_dimension,
            physical_dimension=physical_dimension,
            train_end_index=train_end_index,
            valid_end_index=valid_end_index,
            physical_data_path=physical_data_path,
            native_data_dir=None,
            decoder_run_dir=None,
            native_data_source="physical_data_path",
        )

        logger.debug(
            "Loaded forecast data | mode=%s native=%s/%s/%s physical_dim=%d",
            data_mode,
            tuple(result.native_train.shape),
            tuple(result.native_valid.shape),
            tuple(result.native_test.shape),
            result.physical_dimension,
        )

        return result

    native_data_dir = _resolve_path(
        repo_root,
        require_key(data_cfg, "native_data_dir", "data"),
    )
    decoder_run_dir = _resolve_path(
        repo_root,
        require_key(data_cfg, "decoder_run_dir", "data"),
    )

    if not decoder_run_dir.exists():
        raise FileNotFoundError(
            f"Decoder run directory does not exist: {decoder_run_dir}"
        )

    native_train, native_valid, native_test, native_data_source = _load_latent_data(
        native_data_dir=native_data_dir,
        train_end_index=train_end_index,
        valid_end_index=valid_end_index,
        expected_total_steps=int(physical_series.shape[0]),
        expected_train_steps=int(physical_train.shape[0]),
        expected_valid_steps=int(physical_valid.shape[0]),
        expected_test_steps=int(physical_test.shape[0]),
    )

    native_dimension = _validate_same_feature_dimension(
        [native_train, native_valid, native_test],
        names=["native_train", "native_valid", "native_test"],
    )

    if native_dimension != expected_native_dimension:
        raise ValueError(
            f"Native dimension mismatch for data_mode='{data_mode}': "
            f"expected {expected_native_dimension}, got {native_dimension}."
        )

    _validate_matching_split_lengths(
        native_train=native_train,
        native_valid=native_valid,
        native_test=native_test,
        physical_train=physical_train,
        physical_valid=physical_valid,
        physical_test=physical_test,
    )

    result = ForecastData(
        data_mode=data_mode,
        native_train=native_train,
        native_valid=native_valid,
        native_test=native_test,
        physical_train=physical_train,
        physical_valid=physical_valid,
        physical_test=physical_test,
        native_dimension=native_dimension,
        physical_dimension=physical_dimension,
        train_end_index=train_end_index,
        valid_end_index=valid_end_index,
        physical_data_path=physical_data_path,
        native_data_dir=native_data_dir,
        decoder_run_dir=decoder_run_dir,
        native_data_source=native_data_source,
    )

    logger.debug(
        "Loaded forecast data | mode=%s source=%s native=%s/%s/%s physical=%s/%s/%s",
        data_mode,
        native_data_source,
        tuple(result.native_train.shape),
        tuple(result.native_valid.shape),
        tuple(result.native_test.shape),
        tuple(result.physical_train.shape),
        tuple(result.physical_valid.shape),
        tuple(result.physical_test.shape),
    )

    return result
