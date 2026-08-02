from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)

Array = np.ndarray


@dataclass(frozen=True)
class ForecastErrors:
    """
    Per-timestep relative-L2 error curve over the evaluated forecast window.

    The first skip_initial timesteps are usually seed/context values used to
    start an autonomous rollout. They are not scored.
    """

    relative_l2: Array
    skip_initial: int
    dt: float

    @property
    def n_eval_steps(self) -> int:
        return int(len(self.relative_l2))

    @property
    def eval_index(self) -> Array:
        return np.arange(self.n_eval_steps, dtype=np.int64)

    @property
    def original_index(self) -> Array:
        return self.eval_index + int(self.skip_initial)

    @property
    def time(self) -> Array:
        return self.eval_index.astype(np.float64) * float(self.dt)


def _as_2d_float_array(x: Array, *, name: str) -> Array:
    """
    Convert input to a 2D float64 array with shape:
        (time_steps, state_dimension)
    """
    arr = np.asarray(x, dtype=np.float64)

    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D array with shape (time_steps, dimension), "
            f"got shape {arr.shape}."
        )

    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one timestep.")

    if arr.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one state dimension.")

    return arr


def _require_positive_float(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive float, got bool.")

    value = float(value)

    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}.")

    return value


def _require_nonnegative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative integer, got bool.")

    value = int(value)

    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")

    return value


def find_first_nonfinite_row(x: Array) -> int | None:
    """
    Return the first timestep where any state dimension is NaN or Inf.
    """
    x = _as_2d_float_array(x, name="x")

    row_is_finite = np.all(np.isfinite(x), axis=1)
    bad = np.flatnonzero(~row_is_finite)

    return int(bad[0]) if bad.size else None


def find_first_abs_exceedance_row(x: Array, *, max_abs: float) -> int | None:
    """
    Return the first timestep where max(abs(state)) exceeds max_abs.

    This detects finite but physically/numerically exploded predictions.
    """
    max_abs = _require_positive_float(max_abs, name="max_abs")
    x = _as_2d_float_array(x, name="x")

    with np.errstate(over="ignore", invalid="ignore"):
        row_exceeds = np.any(np.abs(x) > max_abs, axis=1)

    bad = np.flatnonzero(row_exceeds)
    return int(bad[0]) if bad.size else None


def _first_existing_index(*indices: int | None) -> int | None:
    valid = [idx for idx in indices if idx is not None]
    return min(valid) if valid else None


def validate_true_trajectory(y_true: Array) -> Array:
    """
    Validate true/reference data.

    True data must be finite. If the reference trajectory contains NaN/Inf,
    the dataset or split is invalid and the experiment should fail.
    """
    y_true = _as_2d_float_array(y_true, name="y_true")
    first_bad = find_first_nonfinite_row(y_true)

    if first_bad is not None:
        raise ValueError(
            f"y_true contains NaN or Inf at timestep {first_bad}. "
            "Reference data must be finite."
        )

    return y_true


def validate_prediction_shape(
    y_pred: Array,
    *,
    expected_shape: tuple[int, int],
) -> Array:
    """
    Validate prediction array shape.

    Prediction values may contain NaN/Inf because numerical forecast failure is
    handled as part of forecast evaluation.
    """
    y_pred = _as_2d_float_array(y_pred, name="y_pred")

    if y_pred.shape != expected_shape:
        raise ValueError(
            f"Prediction shape mismatch: expected {expected_shape}, got {y_pred.shape}."
        )

    return y_pred


def mask_prediction_after_first_unsafe(
    y_pred: Array,
    *,
    safety_max_abs: float,
) -> tuple[Array, dict[str, Any]]:
    """
    Replace prediction rows from the first unsafe timestep onward with NaN.

    A prediction timestep is unsafe if:
      1. any state component is NaN or Inf, OR
      2. abs(any state component) > safety_max_abs.

    This keeps the metric honest: once the rollout fails numerically, all later
    timesteps are treated as invalid.
    """
    safety_max_abs = _require_positive_float(
        safety_max_abs,
        name="safety_max_abs",
    )

    y_pred = _as_2d_float_array(y_pred, name="y_pred")

    first_nonfinite = find_first_nonfinite_row(y_pred)
    first_exploded = find_first_abs_exceedance_row(
        y_pred,
        max_abs=safety_max_abs,
    )
    first_unsafe = _first_existing_index(first_nonfinite, first_exploded)

    y_safe = y_pred.astype(np.float64, copy=True)

    if first_unsafe is not None:
        y_safe[first_unsafe:] = np.nan
        logger.debug(
            "Masked unsafe prediction | first_nonfinite=%s first_exploded=%s "
            "first_unsafe=%s safety_max_abs=%s",
            first_nonfinite,
            first_exploded,
            first_unsafe,
            safety_max_abs,
        )

    report = {
        "prediction_safety_max_abs": float(safety_max_abs),
        "prediction_first_nonfinite_index": first_nonfinite,
        "prediction_first_exploded_index": first_exploded,
        "prediction_first_unsafe_index": first_unsafe,
        "prediction_masked_from_index": first_unsafe,
    }

    return y_safe, report


def compute_relative_l2_error(
    y_true: Array,
    y_pred: Array,
    *,
    eps: float = 1e-12,
) -> Array:
    """
    Compute relative-L2 error per timestep.

    Formula:
        relative_l2(t) =
            ||y_true(t) - y_pred(t)||_2 / (||y_true(t)||_2 + eps)

    This is the final thesis metric used consistently across forecasting
    models.
    """
    eps = _require_positive_float(eps, name="eps")

    y_true = validate_true_trajectory(y_true)
    y_pred = validate_prediction_shape(y_pred, expected_shape=y_true.shape)

    with np.errstate(over="ignore", invalid="ignore"):
        numerator = np.linalg.norm(y_true - y_pred, axis=1)
        denominator = np.linalg.norm(y_true, axis=1) + eps
        relative_l2 = numerator / denominator

    return relative_l2.astype(np.float64)


def _finite_stat(x: Array, stat: str) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]

    if finite.size == 0:
        return None

    if stat == "mean":
        return float(np.mean(finite))

    if stat == "median":
        return float(np.median(finite))

    if stat == "max":
        return float(np.max(finite))

    if stat == "final":
        return float(x[-1]) if np.isfinite(x[-1]) else None

    raise ValueError(f"Unknown statistic: {stat}")


def compute_relative_l2_horizon(
    relative_l2: Array,
    *,
    threshold: float,
    dt: float,
    skip_initial: int,
) -> dict[str, Any]:
    """
    Compute forecast horizon from the relative-L2 error curve.

    Failure condition:
        relative_l2 is NaN/Inf OR relative_l2 > threshold

    The horizon is the first evaluated timestep where failure occurs. If no
    failure occurs, the horizon reaches the end of the evaluated window.
    """
    threshold = _require_positive_float(threshold, name="threshold")
    dt = _require_positive_float(dt, name="dt")
    skip_initial = _require_nonnegative_int(skip_initial, name="skip_initial")

    relative_l2 = np.asarray(relative_l2, dtype=np.float64)

    if relative_l2.ndim != 1:
        raise ValueError(f"relative_l2 must be 1D, got shape {relative_l2.shape}.")

    if relative_l2.size == 0:
        raise ValueError("relative_l2 must contain at least one value.")

    failure_mask = (~np.isfinite(relative_l2)) | (relative_l2 > threshold)
    failure_indices = np.flatnonzero(failure_mask)

    if failure_indices.size == 0:
        first_failure_eval_index = None
        first_failure_original_index = None
        first_failure_time = None
        first_failure_reason = None
        first_failure_relative_l2 = None
        horizon_steps = int(relative_l2.size)
        reached_end = True
    else:
        first_failure_eval_index = int(failure_indices[0])
        first_failure_original_index = int(first_failure_eval_index + skip_initial)
        first_failure_time = float(first_failure_eval_index * dt)
        horizon_steps = first_failure_eval_index
        reached_end = False

        if not np.isfinite(relative_l2[first_failure_eval_index]):
            first_failure_reason = "nonfinite_relative_l2"
            first_failure_relative_l2 = None
        else:
            first_failure_reason = "threshold_exceeded"
            first_failure_relative_l2 = float(relative_l2[first_failure_eval_index])

    return {
        "relative_l2_threshold": float(threshold),
        "relative_l2_horizon_steps": int(horizon_steps),
        "relative_l2_horizon_time": float(horizon_steps * dt),
        "relative_l2_horizon_reached_end": bool(reached_end),
        "relative_l2_first_failure_eval_index": first_failure_eval_index,
        "relative_l2_first_failure_original_index": first_failure_original_index,
        "relative_l2_first_failure_time": first_failure_time,
        "relative_l2_first_failure_reason": first_failure_reason,
        "relative_l2_first_failure_value": first_failure_relative_l2,
    }


def evaluate_forecast(
    y_true: Array,
    y_pred: Array,
    *,
    dt: float,
    relative_l2_threshold: float,
    skip_initial: int,
    prediction_safety_max_abs: float,
) -> tuple[dict[str, Any], ForecastErrors]:
    """
    Evaluate a forecast using the final thesis metric: relative-L2 error.

    Safety logic:
      1. true data must be finite
      2. prediction NaN/Inf is detected
      3. prediction explosion is detected using prediction_safety_max_abs
      4. prediction is masked with NaN from first unsafe timestep onward
      5. relative-L2 is computed
      6. horizon is the first timestep where relative-L2 is nonfinite or above
         threshold
    """
    dt = _require_positive_float(dt, name="dt")
    relative_l2_threshold = _require_positive_float(
        relative_l2_threshold,
        name="relative_l2_threshold",
    )
    prediction_safety_max_abs = _require_positive_float(
        prediction_safety_max_abs,
        name="prediction_safety_max_abs",
    )
    skip_initial = _require_nonnegative_int(skip_initial, name="skip_initial")

    y_true = validate_true_trajectory(y_true)
    y_pred = validate_prediction_shape(y_pred, expected_shape=y_true.shape)

    if skip_initial >= y_true.shape[0]:
        raise ValueError(
            f"skip_initial={skip_initial} is too large for trajectory length "
            f"{y_true.shape[0]}."
        )

    y_pred_safe, safety_report = mask_prediction_after_first_unsafe(
        y_pred,
        safety_max_abs=prediction_safety_max_abs,
    )

    y_true_eval = y_true[skip_initial:]
    y_pred_eval = y_pred_safe[skip_initial:]

    relative_l2 = compute_relative_l2_error(y_true_eval, y_pred_eval)

    horizon = compute_relative_l2_horizon(
        relative_l2,
        threshold=relative_l2_threshold,
        dt=dt,
        skip_initial=skip_initial,
    )

    errors = ForecastErrors(
        relative_l2=relative_l2,
        skip_initial=int(skip_initial),
        dt=float(dt),
    )

    finite_relative_l2_fraction = float(np.mean(np.isfinite(relative_l2)))

    metrics = {
        "metric_name": "relative_l2",
        "n_total_steps": int(y_true.shape[0]),
        "n_eval_steps": int(y_true_eval.shape[0]),
        "state_dimension": int(y_true.shape[1]),
        "dt": float(dt),
        "skip_initial": int(skip_initial),
        **safety_report,
        "finite_relative_l2_fraction": finite_relative_l2_fraction,
        "relative_l2_mean": _finite_stat(relative_l2, "mean"),
        "relative_l2_median": _finite_stat(relative_l2, "median"),
        "relative_l2_max": _finite_stat(relative_l2, "max"),
        "relative_l2_final": _finite_stat(relative_l2, "final"),
        **horizon,
    }

    logger.debug(
        "Evaluated forecast | steps=%d eval_steps=%d dim=%d horizon_time=%s "
        "mean_l2=%s first_failure=%s",
        y_true.shape[0],
        y_true_eval.shape[0],
        y_true.shape[1],
        metrics["relative_l2_horizon_time"],
        metrics["relative_l2_mean"],
        metrics["relative_l2_first_failure_reason"],
    )

    return metrics, errors


def save_error_curve_csv(path: str | Path, errors: ForecastErrors) -> None:
    """
    Save the evaluated relative-L2 error curve for plotting and inspection.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["eval_index", "original_index", "time", "relative_l2"])

        for eval_idx, original_idx, time, rel in zip(
            errors.eval_index,
            errors.original_index,
            errors.time,
            errors.relative_l2,
        ):
            writer.writerow([
                int(eval_idx),
                int(original_idx),
                float(time),
                float(rel) if np.isfinite(rel) else "nan",
            ])

    logger.debug(
        "Saved relative-L2 error curve | path=%s rows=%d",
        path,
        errors.n_eval_steps,
    )
