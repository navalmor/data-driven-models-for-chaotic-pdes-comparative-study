from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


logger = logging.getLogger(__name__)

Array = np.ndarray
QuadraticMode = Literal["diagonal", "pairwise"]


def _require_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer, got bool.")

    value_int = int(value)

    if value_int != value:
        raise TypeError(f"{name} must be an integer, got {value!r}.")

    if value_int <= 0:
        raise ValueError(f"{name} must be positive, got {value_int}.")

    return value_int


def _require_positive_float(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive float, got bool.")

    value_float = float(value)

    if not np.isfinite(value_float) or value_float <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value_float}.")

    return value_float


@dataclass(frozen=True)
class NGRCParameters:
    """
    Hyperparameters for Next-Generation Reservoir Computing.

    context_steps:
        Number of delayed states used as context.

    delay_spacing:
        Spacing between delayed states. For example, context_steps=3 and
        delay_spacing=2 uses [x[t-5], x[t-3], x[t-1]] to predict x[t].

    ridge_alpha:
        Ridge regularization strength for the linear readout.

    include_*:
        Controls which nonlinear feature families are included.

    quadratic_mode:
        "diagonal" uses elementwise squares only.
        "pairwise" uses all unique pairwise quadratic monomials.
    """

    context_steps: int
    delay_spacing: int = 1
    ridge_alpha: float = 1e-6

    include_linear: bool = True
    include_quadratic: bool = True
    quadratic_mode: QuadraticMode = "diagonal"

    include_sin: bool = False
    include_cos: bool = False
    include_bias: bool = False

    def validate(self) -> None:
        _require_positive_int(self.context_steps, name="context_steps")
        _require_positive_int(self.delay_spacing, name="delay_spacing")
        _require_positive_float(self.ridge_alpha, name="ridge_alpha")

        if self.quadratic_mode not in {"diagonal", "pairwise"}:
            raise ValueError(
                f"quadratic_mode must be 'diagonal' or 'pairwise', "
                f"got {self.quadratic_mode!r}."
            )

        feature_flags = [
            self.include_linear,
            self.include_quadratic,
            self.include_sin,
            self.include_cos,
            self.include_bias,
        ]

        if not all(isinstance(flag, bool) for flag in feature_flags):
            raise TypeError("All include_* feature flags must be boolean values.")

        if not any(feature_flags):
            raise ValueError("At least one feature family must be enabled.")

    @property
    def required_history_length(self) -> int:
        """
        Number of consecutive timesteps required to build one delayed context.

        For delay_spacing=1:
            required_history_length = context_steps

        For context_steps=3, delay_spacing=2:
            required_history_length = 5
            because we need x[t-5], x[t-3], x[t-1] to predict x[t].
        """
        return (int(self.context_steps) - 1) * int(self.delay_spacing) + 1


class NGRCModel:
    """
    Next-Generation Reservoir Computing model.

    The model learns:
        nonlinear_features(delayed_context) @ weights ≈ next_state

    This class is intentionally model-only. It does not load datasets,
    standardize data, decode latent trajectories, compute horizons, or write
    experiment summaries.
    """

    def __init__(self, parameters: NGRCParameters):
        parameters.validate()
        self.parameters = parameters

        self.weights: Array | None = None
        self.input_dimension: int | None = None
        self.output_dimension: int | None = None
        self.feature_dimension: int | None = None

    @property
    def is_fitted(self) -> bool:
        return self.weights is not None

    @property
    def required_history_length(self) -> int:
        return self.parameters.required_history_length

    def _require_fitted(self) -> None:
        if self.weights is None:
            raise RuntimeError("NGRCModel is not fitted. Call fit(...) or load(...).")

    @staticmethod
    def _validate_2d_array(x: Array, *, name: str, require_finite: bool) -> Array:
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
            raise ValueError(f"{name} contains NaN or Inf.")

        return arr

    def _delayed_indices_for_target(self, target_index: int) -> list[int]:
        """
        Return delayed input indices used to predict x[target_index].

        The indices are returned in chronological order from oldest to newest.
        """
        k = int(self.parameters.context_steps)
        s = int(self.parameters.delay_spacing)

        return [
            target_index - 1 - (k - 1 - j) * s
            for j in range(k)
        ]

    def _delayed_vector_from_history(self, history: Array) -> Array:
        """
        Build one flattened delayed vector from a recent history block.

        history must have shape:
            (required_history_length, state_dimension)
        """
        history = self._validate_2d_array(
            history,
            name="history",
            require_finite=False,
        )

        expected_steps = self.required_history_length

        if history.shape[0] != expected_steps:
            raise ValueError(
                f"history must contain exactly {expected_steps} timesteps, "
                f"got {history.shape[0]}."
            )

        k = int(self.parameters.context_steps)
        s = int(self.parameters.delay_spacing)
        last = history.shape[0] - 1

        delayed_indices = [
            last - (k - 1 - j) * s
            for j in range(k)
        ]

        return history[delayed_indices].reshape(-1).astype(np.float64)

    def _make_delayed_matrix(self, sequence: Array) -> Array:
        """
        Build the delayed input matrix for training.

        For a sequence with T timesteps, number of training samples is:
            T - required_history_length
        """
        sequence = self._validate_2d_array(
            sequence,
            name="sequence",
            require_finite=True,
        )

        history_length = self.required_history_length
        n_steps, state_dim = sequence.shape

        if n_steps <= history_length:
            raise ValueError(
                f"sequence length must be greater than required_history_length. "
                f"Got sequence length {n_steps}, required_history_length {history_length}."
            )

        n_samples = n_steps - history_length
        flat_dim = int(self.parameters.context_steps) * int(state_dim)

        delayed = np.empty((n_samples, flat_dim), dtype=np.float64)

        for row, target_index in enumerate(range(history_length, n_steps)):
            indices = self._delayed_indices_for_target(target_index)
            delayed[row] = sequence[indices].reshape(-1)

        return delayed

    def _feature_dimension_from_flat_dimension(self, flat_dim: int) -> int:
        flat_dim = _require_positive_int(flat_dim, name="flat_dim")
        p = self.parameters
        dim = 0

        if p.include_bias:
            dim += 1

        if p.include_linear:
            dim += flat_dim

        if p.include_quadratic:
            if p.quadratic_mode == "diagonal":
                dim += flat_dim
            elif p.quadratic_mode == "pairwise":
                dim += flat_dim * (flat_dim + 1) // 2
            else:
                raise ValueError(f"Unsupported quadratic_mode={p.quadratic_mode!r}.")

        if p.include_sin:
            dim += flat_dim

        if p.include_cos:
            dim += flat_dim

        return int(dim)

    def _features_from_delayed_matrix(
        self,
        delayed: Array,
        *,
        require_finite: bool,
    ) -> Array:
        """
        Convert delayed vectors into NGRC feature vectors.

        During training, require_finite=True.
        During autonomous rollout, require_finite=False so NaN/Inf can propagate
        after numerical failure and be handled later by metrics.py.
        """
        delayed = self._validate_2d_array(
            delayed,
            name="delayed",
            require_finite=require_finite,
        )

        p = self.parameters
        parts: list[Array] = []

        with np.errstate(over="ignore", invalid="ignore"):
            if p.include_bias:
                parts.append(np.ones((delayed.shape[0], 1), dtype=np.float64))

            if p.include_linear:
                parts.append(delayed)

            if p.include_quadratic:
                if p.quadratic_mode == "diagonal":
                    parts.append(delayed ** 2)
                elif p.quadratic_mode == "pairwise":
                    i, j = np.triu_indices(delayed.shape[1])
                    parts.append(delayed[:, i] * delayed[:, j])
                else:
                    raise ValueError(
                        f"Unsupported quadratic_mode={p.quadratic_mode!r}."
                    )

            if p.include_sin:
                parts.append(np.sin(delayed))

            if p.include_cos:
                parts.append(np.cos(delayed))

            features = np.concatenate(parts, axis=1)

        if require_finite and not np.all(np.isfinite(features)):
            raise ValueError("NGRC feature matrix contains NaN or Inf.")

        return features.astype(np.float64)

    def _features_from_history(self, history: Array) -> Array:
        delayed = self._delayed_vector_from_history(history)
        delayed_2d = delayed.reshape(1, -1)

        return self._features_from_delayed_matrix(
            delayed_2d,
            require_finite=False,
        )

    def fit(self, x_train: Array) -> "NGRCModel":
        """
        Fit NGRC readout weights using ridge regression.

        x_train shape:
            (time_steps, state_dimension)

        Training pairs:
            delayed context from x_train -> next state
        """
        x_train = self._validate_2d_array(
            x_train,
            name="x_train",
            require_finite=True,
        )

        self.input_dimension = int(x_train.shape[1])
        self.output_dimension = int(x_train.shape[1])

        delayed = self._make_delayed_matrix(x_train)
        features = self._features_from_delayed_matrix(
            delayed,
            require_finite=True,
        )

        y_target = x_train[self.required_history_length:]

        if y_target.shape[0] != features.shape[0]:
            raise RuntimeError(
                "Internal training alignment error: feature rows and targets differ."
            )

        self.feature_dimension = int(features.shape[1])

        expected_feature_dimension = self._feature_dimension_from_flat_dimension(
            flat_dim=int(self.parameters.context_steps) * int(x_train.shape[1])
        )

        if self.feature_dimension != expected_feature_dimension:
            raise RuntimeError(
                f"Feature dimension mismatch: expected {expected_feature_dimension}, "
                f"got {self.feature_dimension}."
            )

        alpha = float(self.parameters.ridge_alpha)

        gram = features.T @ features
        rhs = features.T @ y_target

        regularizer = alpha * np.eye(gram.shape[0], dtype=np.float64)

        # If a bias column is present, it is the first feature column.
        # We do not regularize the bias term, matching standard ridge practice
        # where intercepts are not penalized.
        if self.parameters.include_bias:
            regularizer[0, 0] = 0.0

        system_matrix = gram + regularizer

        try:
            self.weights = np.linalg.solve(system_matrix, rhs)
            solver = "solve"
        except np.linalg.LinAlgError:
            self.weights = np.linalg.lstsq(system_matrix, rhs, rcond=None)[0]
            solver = "lstsq"

        if self.weights.shape != (self.feature_dimension, self.output_dimension):
            raise RuntimeError(
                f"Unexpected weight shape: got {self.weights.shape}, "
                f"expected {(self.feature_dimension, self.output_dimension)}."
            )

        if not np.all(np.isfinite(self.weights)):
            raise ValueError("Fitted NGRC weights contain NaN or Inf.")

        logger.debug(
            "Fitted NGRC | samples=%d input_dim=%d history=%d features=%d "
            "ridge=%s solver=%s",
            features.shape[0],
            self.input_dimension,
            self.required_history_length,
            self.feature_dimension,
            alpha,
            solver,
        )

        return self

    def predict_next(self, history: Array) -> Array:
        """
        Predict one next state from a finite history block.

        history shape:
            (required_history_length, input_dimension)
        """
        self._require_fitted()

        history = self._validate_2d_array(
            history,
            name="history",
            require_finite=True,
        )

        if history.shape != (self.required_history_length, self.input_dimension):
            raise ValueError(
                f"history shape mismatch: expected "
                f"{(self.required_history_length, self.input_dimension)}, "
                f"got {history.shape}."
            )

        features = self._features_from_history(history)

        if features.shape[1] != self.feature_dimension:
            raise RuntimeError(
                f"Feature dimension mismatch during prediction: expected "
                f"{self.feature_dimension}, got {features.shape[1]}."
            )

        with np.errstate(over="ignore", invalid="ignore"):
            pred = features @ self.weights

        return pred.reshape(-1).astype(np.float64)

    def _predict_next_allowing_nonfinite(self, history: Array) -> Array:
        """
        Internal rollout prediction.

        This does not require finite history because once a rollout has become
        numerically invalid, NaN/Inf should propagate into the prediction array
        rather than crashing the experiment.
        """
        self._require_fitted()

        features = self._features_from_history(history)

        with np.errstate(over="ignore", invalid="ignore"):
            pred = features @ self.weights

        return pred.reshape(-1).astype(np.float64)

    def predict_autoregressive(
        self,
        *,
        initial_context: Array,
        n_steps: int,
    ) -> Array:
        """
        Run autonomous closed-loop prediction.

        initial_context shape:
            (required_history_length, input_dimension)

        Output shape:
            (n_steps, output_dimension)

        The first required_history_length rows are copied from initial_context.
        Model-generated predictions start after that initial context.
        """
        self._require_fitted()

        n_steps = _require_positive_int(n_steps, name="n_steps")
        history_length = self.required_history_length

        if n_steps < history_length:
            raise ValueError(
                f"n_steps must be at least required_history_length={history_length}, "
                f"got {n_steps}."
            )

        initial_context = self._validate_2d_array(
            initial_context,
            name="initial_context",
            require_finite=True,
        )

        expected_shape = (history_length, self.input_dimension)
        if initial_context.shape != expected_shape:
            raise ValueError(
                f"initial_context shape mismatch: expected {expected_shape}, "
                f"got {initial_context.shape}."
            )

        pred = np.empty((n_steps, self.output_dimension), dtype=np.float64)
        pred[:history_length] = initial_context

        first_nonfinite_index: int | None = None

        for t in range(history_length, n_steps):
            history = pred[t - history_length:t]
            pred[t] = self._predict_next_allowing_nonfinite(history)

            if not np.all(np.isfinite(pred[t])):
                first_nonfinite_index = int(t)
                pred[t + 1:] = np.nan
                break

        logger.debug(
            "Completed NGRC rollout | n_steps=%d history=%d first_nonfinite=%s",
            n_steps,
            history_length,
            first_nonfinite_index,
        )

        return pred

    def save(self, path: str | Path) -> None:
        """
        Save fitted NGRC model weights and architecture metadata.

        A saved model can later be loaded and used for prediction without
        retraining.
        """
        self._require_fitted()

        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        p = self.parameters

        np.savez_compressed(
            path,
            weights=self.weights,
            context_steps=np.array(p.context_steps, dtype=np.int64),
            delay_spacing=np.array(p.delay_spacing, dtype=np.int64),
            ridge_alpha=np.array(p.ridge_alpha, dtype=np.float64),
            include_linear=np.array(p.include_linear, dtype=np.bool_),
            include_quadratic=np.array(p.include_quadratic, dtype=np.bool_),
            quadratic_mode=np.array(p.quadratic_mode),
            include_sin=np.array(p.include_sin, dtype=np.bool_),
            include_cos=np.array(p.include_cos, dtype=np.bool_),
            include_bias=np.array(p.include_bias, dtype=np.bool_),
            input_dimension=np.array(self.input_dimension, dtype=np.int64),
            output_dimension=np.array(self.output_dimension, dtype=np.int64),
            feature_dimension=np.array(self.feature_dimension, dtype=np.int64),
            required_history_length=np.array(
                self.required_history_length,
                dtype=np.int64,
            ),
        )

        logger.debug(
            "Saved NGRC model | path=%s input_dim=%d output_dim=%d features=%d",
            path,
            self.input_dimension,
            self.output_dimension,
            self.feature_dimension,
        )

    @classmethod
    def load(cls, path: str | Path) -> "NGRCModel":
        """
        Load a fitted NGRC model from disk.
        """
        path = Path(path).expanduser()

        if not path.exists():
            raise FileNotFoundError(f"NGRC model file does not exist: {path}")

        with np.load(path, allow_pickle=False) as data:
            parameters = NGRCParameters(
                context_steps=int(data["context_steps"]),
                delay_spacing=int(data["delay_spacing"]),
                ridge_alpha=float(data["ridge_alpha"]),
                include_linear=bool(data["include_linear"]),
                include_quadratic=bool(data["include_quadratic"]),
                quadratic_mode=str(data["quadratic_mode"].item()),
                include_sin=bool(data["include_sin"]),
                include_cos=bool(data["include_cos"]),
                include_bias=bool(data["include_bias"]),
            )

            model = cls(parameters)

            model.weights = np.asarray(data["weights"], dtype=np.float64)
            model.input_dimension = int(data["input_dimension"])
            model.output_dimension = int(data["output_dimension"])
            model.feature_dimension = int(data["feature_dimension"])

            saved_required_history_length = int(data["required_history_length"])

        if saved_required_history_length != model.required_history_length:
            raise ValueError(
                f"Loaded required_history_length mismatch: metadata has "
                f"{saved_required_history_length}, parameters imply "
                f"{model.required_history_length}."
            )

        if model.weights.ndim != 2:
            raise ValueError(
                f"Loaded weights must be 2D, got shape {model.weights.shape}."
            )

        expected_shape = (model.feature_dimension, model.output_dimension)
        if model.weights.shape != expected_shape:
            raise ValueError(
                f"Loaded weight shape mismatch: expected {expected_shape}, "
                f"got {model.weights.shape}."
            )

        if model.input_dimension <= 0 or model.output_dimension <= 0:
            raise ValueError(
                f"Loaded model dimensions must be positive, got "
                f"input_dimension={model.input_dimension}, "
                f"output_dimension={model.output_dimension}."
            )

        expected_feature_dimension = model._feature_dimension_from_flat_dimension(
            flat_dim=int(model.parameters.context_steps) * int(model.input_dimension)
        )

        if model.feature_dimension != expected_feature_dimension:
            raise ValueError(
                f"Loaded feature dimension mismatch: expected "
                f"{expected_feature_dimension}, got {model.feature_dimension}."
            )

        if not np.all(np.isfinite(model.weights)):
            raise ValueError(f"Loaded NGRC weights contain NaN or Inf: {path}")

        logger.debug(
            "Loaded NGRC model | path=%s input_dim=%d output_dim=%d features=%d",
            path,
            model.input_dimension,
            model.output_dimension,
            model.feature_dimension,
        )

        return model
