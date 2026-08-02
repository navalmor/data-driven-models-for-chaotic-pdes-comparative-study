from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)

Array = np.ndarray


def _as_2d_float_array(
    x: Array,
    *,
    name: str,
    require_finite: bool,
) -> Array:
    """
    Convert input to a 2D float64 NumPy array.

    Forecasting data is expected to have shape:
        (time_steps, state_dimension)

    Training/validation/test truth data should be finite. Model predictions may
    contain NaN/Inf after numerical divergence, so some prediction-side calls
    intentionally allow non-finite values to pass through for later evaluation.
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

    if require_finite and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    return arr


@dataclass(frozen=True)
class Standardizer:
    """
    Train-only mean/std standardizer.

    The standardizer must be fitted on training data only and then reused for
    validation data, test data, and model predictions.

    Formula:
        x_scaled = (x - mean) / scale
        x        = x_scaled * scale + mean
    """

    mean: Array
    scale: Array
    eps: float = 1e-12

    @classmethod
    def fit(cls, x_train: Array, *, eps: float = 1e-12) -> "Standardizer":
        """
        Fit mean and standard deviation from training data only.
        """
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}.")

        x_train = _as_2d_float_array(
            x_train,
            name="x_train",
            require_finite=True,
        )

        mean = x_train.mean(axis=0)
        raw_scale = x_train.std(axis=0)

        # Avoid division by zero for constant dimensions.
        constant_mask = raw_scale < eps
        scale = np.where(constant_mask, 1.0, raw_scale)

        standardizer = cls(mean=mean, scale=scale, eps=eps)
        standardizer._validate()

        logger.debug(
            "Fitted standardizer | samples=%d dimension=%d constant_dimensions=%d",
            x_train.shape[0],
            x_train.shape[1],
            int(np.sum(constant_mask)),
        )

        return standardizer

    @property
    def n_features(self) -> int:
        return int(np.asarray(self.mean).shape[0])

    def _validate(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)

        if mean.ndim != 1:
            raise ValueError(f"mean must be 1D, got shape {mean.shape}.")

        if scale.ndim != 1:
            raise ValueError(f"scale must be 1D, got shape {scale.shape}.")

        if mean.shape != scale.shape:
            raise ValueError(
                f"mean and scale must have same shape, got {mean.shape} and {scale.shape}."
            )

        if mean.shape[0] == 0:
            raise ValueError("mean/scale must contain at least one feature.")

        if not np.all(np.isfinite(mean)):
            raise ValueError("mean contains NaN or infinite values.")

        if not np.all(np.isfinite(scale)):
            raise ValueError("scale contains NaN or infinite values.")

        if np.any(scale <= 0.0):
            raise ValueError("scale must be strictly positive.")

        if float(self.eps) <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}.")

    def transform(self, x: Array, *, name: str = "x") -> Array:
        """
        Standardize finite data using stored training statistics.

        This is used for true train/validation/test data, so non-finite values
        are rejected.
        """
        self._validate()

        x = _as_2d_float_array(
            x,
            name=name,
            require_finite=True,
        )

        if x.shape[1] != self.n_features:
            raise ValueError(
                f"{name} has dimension {x.shape[1]}, but standardizer expects "
                f"{self.n_features}."
            )

        return ((x - self.mean) / self.scale).astype(np.float64)

    def inverse_transform(
        self,
        x_scaled: Array,
        *,
        name: str = "x_scaled",
        require_finite: bool = False,
    ) -> Array:
        """
        Convert standardized data back to original scale.

        By default, non-finite values are allowed because model predictions may
        contain NaN/Inf after a divergent rollout. Those values should propagate
        to metrics.py, where the first failed forecast index is measured.
        """
        self._validate()

        x_scaled = _as_2d_float_array(
            x_scaled,
            name=name,
            require_finite=require_finite,
        )

        if x_scaled.shape[1] != self.n_features:
            raise ValueError(
                f"{name} has dimension {x_scaled.shape[1]}, but standardizer expects "
                f"{self.n_features}."
            )

        with np.errstate(over="ignore", invalid="ignore"):
            return (x_scaled * self.scale + self.mean).astype(np.float64)

    def save(self, path: str | Path) -> None:
        """
        Save standardization statistics to an .npz file.
        """
        self._validate()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            mean=np.asarray(self.mean, dtype=np.float64),
            scale=np.asarray(self.scale, dtype=np.float64),
            eps=np.asarray(self.eps, dtype=np.float64),
        )

        logger.debug(
            "Saved standardizer | path=%s dimension=%d",
            path,
            self.n_features,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Standardizer":
        """
        Load standardization statistics from an .npz file.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Standardizer file does not exist: {path}")

        with np.load(path) as data:
            mean = data["mean"].astype(np.float64)
            scale = data["scale"].astype(np.float64)
            eps = float(np.asarray(data["eps"]).item()) if "eps" in data.files else 1e-12

        standardizer = cls(mean=mean, scale=scale, eps=eps)
        standardizer._validate()

        logger.debug(
            "Loaded standardizer | path=%s dimension=%d",
            path,
            standardizer.n_features,
        )

        return standardizer
