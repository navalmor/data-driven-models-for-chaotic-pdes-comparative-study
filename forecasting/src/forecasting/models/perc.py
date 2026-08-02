from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class PERCParameters:
    """
    Echo State Network / Reservoir Computing parameters.

    The reservoir and input weights are fixed random features. Only the linear
    readout is trained by ridge regression.

    State update:
        r_{t+1} = (1 - a) r_t + a tanh(W_in [1, u_t] + W r_t)

    Readout:
        u_{t+1} = W_out phi(u_t, r_{t+1})

    where phi may contain a bias, the current input, and the reservoir state.
    """

    reservoir_size: int
    sparsity: float
    spectral_radius: float
    input_scaling: float
    leaking_rate: float
    ridge_alpha: float
    washout_steps: int
    include_bias: bool = True
    include_input_skip: bool = True
    random_seed: int = 0

    def validate(self) -> None:
        if self.reservoir_size <= 0:
            raise ValueError(f"reservoir_size must be positive, got {self.reservoir_size}.")
        if not (0.0 < self.sparsity <= 1.0):
            raise ValueError(f"sparsity must be in (0, 1], got {self.sparsity}.")
        if self.spectral_radius <= 0.0:
            raise ValueError(f"spectral_radius must be positive, got {self.spectral_radius}.")
        if self.input_scaling <= 0.0:
            raise ValueError(f"input_scaling must be positive, got {self.input_scaling}.")
        if not (0.0 < self.leaking_rate <= 1.0):
            raise ValueError(f"leaking_rate must be in (0, 1], got {self.leaking_rate}.")
        if self.ridge_alpha < 0.0:
            raise ValueError(f"ridge_alpha must be non-negative, got {self.ridge_alpha}.")
        if self.washout_steps < 0:
            raise ValueError(f"washout_steps must be non-negative, got {self.washout_steps}.")
        if not self.include_bias and not self.include_input_skip:
            # This is not mathematically illegal, but it is a poor default for
            # forecasting because the readout would see only reservoir states.
            pass


class _PERCReservoirBackbone:
    """
    Thesis-safe ESN/RC core model.

    This class is intentionally independent from file layouts, metrics, decoding,
    registry, and experiment runners. The runner will handle train-only
    standardization, physical-space metrics, latent decoding, and registry control.
    """

    def __init__(self, parameters: PERCParameters):
        parameters.validate()

        self.parameters = parameters

        self.input_dim: int | None = None
        self.feature_dimension: int | None = None

        self.win: np.ndarray | None = None
        self.w_res: sp.csr_matrix | None = None
        self.w_out: np.ndarray | None = None

        self.pre_scale_spectral_radius: float | None = None
        self.reservoir_scale_factor: float | None = None
        self.actual_spectral_radius: float | None = None

    @property
    def is_initialized(self) -> bool:
        return self.win is not None and self.w_res is not None

    @property
    def is_fit(self) -> bool:
        return self.is_initialized and self.w_out is not None

    @property
    def required_history_length(self) -> int:
        """Number of teacher-forced split samples needed before first forecast."""
        return max(1, int(self.parameters.washout_steps))

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.parameters.random_seed)

    def _estimate_spectral_radius(self, matrix: sp.csr_matrix) -> float:
        if matrix.shape[0] == 1:
            return float(abs(matrix.toarray()[0, 0]))

        try:
            # ARPACK can otherwise choose an internal/random starting vector,
            # which makes the estimated radius slightly non-reproducible even
            # when the reservoir seed is fixed. Use a deterministic v0 so that
            # identical configs produce identical scaled reservoirs.
            rng = self._rng()
            v0 = rng.normal(size=matrix.shape[0])
            v0_norm = np.linalg.norm(v0)
            if v0_norm == 0.0:
                v0 = np.ones(matrix.shape[0], dtype=np.float64)
                v0_norm = np.linalg.norm(v0)
            v0 = v0 / v0_norm

            eigenvalues = eigs(
                matrix,
                k=1,
                which="LM",
                return_eigenvectors=False,
                maxiter=max(1000, 10 * matrix.shape[0]),
                v0=v0,
            )
            radius = float(np.max(np.abs(eigenvalues)))
        except Exception:
            # Robust fallback. This estimates growth by repeated multiplication.
            # It is less exact than eigs for non-normal matrices but avoids a hard
            # failure for difficult random reservoirs.
            rng = self._rng()
            vector = rng.normal(size=matrix.shape[0])
            vector /= np.linalg.norm(vector)

            radius = 0.0
            for _ in range(100):
                new_vector = matrix @ vector
                norm = float(np.linalg.norm(new_vector))
                if norm == 0.0:
                    radius = 0.0
                    break
                vector = new_vector / norm
                radius = norm

        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("Could not estimate a positive finite reservoir spectral radius.")

        return radius

    def initialize(self, input_dim: int) -> None:
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")

        p = self.parameters
        rng = self._rng()

        self.input_dim = int(input_dim)

        self.win = rng.uniform(
            low=-p.input_scaling,
            high=p.input_scaling,
            size=(p.reservoir_size, self.input_dim + 1),
        ).astype(np.float64)

        # Sparse random reservoir with values in [-1, 1].
        # The fixed seed makes hyperparameter comparisons reproducible.
        sparse_rng = np.random.default_rng(p.random_seed + 1)
        w_raw = sp.random(
            p.reservoir_size,
            p.reservoir_size,
            density=p.sparsity,
            format="csr",
            random_state=sparse_rng,
            data_rvs=lambda size: rng.uniform(-1.0, 1.0, size=size),
        ).astype(np.float64)

        pre_radius = self._estimate_spectral_radius(w_raw)
        scale = p.spectral_radius / pre_radius

        self.w_res = (w_raw * scale).tocsr()

        self.pre_scale_spectral_radius = pre_radius
        self.reservoir_scale_factor = scale
        self.actual_spectral_radius = p.spectral_radius

        self.feature_dimension = (
            (1 if p.include_bias else 0)
            + (self.input_dim if p.include_input_skip else 0)
            + p.reservoir_size
        )

    def _require_initialized(self) -> None:
        if self.input_dim is None or self.win is None or self.w_res is None:
            raise RuntimeError("_PERCReservoirBackbone is not initialized.")

    def _require_fit(self) -> None:
        self._require_initialized()
        if self.w_out is None:
            raise RuntimeError("_PERCReservoirBackbone is not fit.")

    def _step(self, u: np.ndarray, state: np.ndarray) -> np.ndarray:
        self._require_initialized()

        if u.shape != (self.input_dim,):
            raise ValueError(f"Expected input shape {(self.input_dim,)}, got {u.shape}.")
        if state.shape != (self.parameters.reservoir_size,):
            raise ValueError(
                f"Expected reservoir state shape {(self.parameters.reservoir_size,)}, got {state.shape}."
            )

        augmented_input = np.empty(self.input_dim + 1, dtype=np.float64)
        augmented_input[0] = 1.0
        augmented_input[1:] = u

        preactivation = self.win @ augmented_input + self.w_res @ state
        proposed_state = np.tanh(preactivation)

        a = self.parameters.leaking_rate
        return (1.0 - a) * state + a * proposed_state

    def _features(self, u: np.ndarray, state: np.ndarray) -> np.ndarray:
        self._require_initialized()

        parts = []

        if self.parameters.include_bias:
            parts.append(np.ones(1, dtype=np.float64))

        if self.parameters.include_input_skip:
            parts.append(u.astype(np.float64, copy=False))

        parts.append(state.astype(np.float64, copy=False))

        return np.concatenate(parts)

    def fit(self, x_train: np.ndarray) -> None:
        """
        Fit the linear readout on a standardized/native training trajectory.

        Alignment:
            after consuming x[t], reservoir state r_{t+1} is used with x[t]
            to predict x[t+1].

        Therefore, during rollout, after warmup consumes x[warmup-1], the first
        autonomous prediction is aligned with truth x[warmup].
        """
        x_train = np.asarray(x_train, dtype=np.float64)

        if x_train.ndim != 2:
            raise ValueError(f"x_train must be 2D, got shape {x_train.shape}.")
        first_training_index = max(0, int(self.parameters.washout_steps) - 1)

        if x_train.shape[0] < first_training_index + 2:
            raise ValueError(
                "Training trajectory too short for washout and one-step targets: "
                f"length={x_train.shape[0]}, washout={self.parameters.washout_steps}, "
                f"first_training_index={first_training_index}."
            )

        if not self.is_initialized:
            self.initialize(input_dim=x_train.shape[1])

        if x_train.shape[1] != self.input_dim:
            raise ValueError(f"input_dim mismatch: model={self.input_dim}, train={x_train.shape[1]}.")

        state = np.zeros(self.parameters.reservoir_size, dtype=np.float64)

        features = []
        targets = []

        for t in range(x_train.shape[0] - 1):
            u_t = x_train[t]
            state = self._step(u_t, state)

            if t >= first_training_index:
                features.append(self._features(u_t, state))
                targets.append(x_train[t + 1])

        phi = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)

        if phi.ndim != 2 or y.ndim != 2:
            raise RuntimeError("Internal RC training matrices have invalid shape.")

        regressor = Ridge(
            alpha=self.parameters.ridge_alpha,
            fit_intercept=False,
            copy_X=False,
        )
        regressor.fit(phi, y)

        # sklearn stores coef_ as shape (output_dim, feature_dim).
        self.w_out = regressor.coef_.T.astype(np.float64, copy=True)

    def predict_one_step(self, u: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Consume input u, update the reservoir state, then predict the next state.

        Numerical note:
            Autonomous RC rollouts can diverge for bad hyperparameters. The runner
            handles this by stopping at the first unsafe prediction. We suppress
            low-level overflow warnings here and expose the non-finite result to
            the caller, where it is converted into a masked/NaN suffix.
        """
        self._require_fit()

        with np.errstate(over="ignore", invalid="ignore"):
            state_next = self._step(u, state)
            phi = self._features(u, state_next)
            y_next = phi @ self.w_out

        return y_next.astype(np.float64, copy=False), state_next

    def rollout_from_split(
        self,
        x_split: np.ndarray,
        *,
        warmup_steps: int | None = None,
        forecast_steps: int | None = None,
        prediction_safety_max_abs: float | None = None,
    ) -> np.ndarray:
        """
        Teacher-force the reservoir for warmup_steps samples, then roll out closed loop.

        If warmup_steps = w:
            consumed warmup samples: x[0], ..., x[w-1]
            first prediction aligns with truth x[w]

        Returned shape:
            (forecast_steps, input_dim)

        Safety behavior:
            If prediction_safety_max_abs is provided, the autonomous rollout stops
            at the first non-finite prediction or first prediction with absolute
            value above this bound. The remaining rows are filled with NaN. This
            keeps bad search trials fast, quiet, and honestly marked as unsafe.
        """
        self._require_fit()

        x_split = np.asarray(x_split, dtype=np.float64)

        if x_split.ndim != 2:
            raise ValueError(f"x_split must be 2D, got shape {x_split.shape}.")
        if x_split.shape[1] != self.input_dim:
            raise ValueError(f"input_dim mismatch: model={self.input_dim}, split={x_split.shape[1]}.")

        if warmup_steps is None:
            warmup_steps = self.required_history_length

        warmup_steps = int(warmup_steps)

        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive.")
        if warmup_steps >= x_split.shape[0]:
            raise ValueError(
                f"warmup_steps must be smaller than split length, got warmup={warmup_steps}, "
                f"length={x_split.shape[0]}."
            )

        max_abs = None
        if prediction_safety_max_abs is not None:
            max_abs = float(prediction_safety_max_abs)
            if not math.isfinite(max_abs) or max_abs <= 0.0:
                raise ValueError(
                    "prediction_safety_max_abs must be positive and finite if provided, "
                    f"got {prediction_safety_max_abs!r}."
                )

        max_steps = x_split.shape[0] - warmup_steps

        if forecast_steps is None:
            forecast_steps = max_steps

        forecast_steps = int(forecast_steps)

        if forecast_steps <= 0:
            raise ValueError("forecast_steps must be positive.")
        if forecast_steps > max_steps:
            raise ValueError(
                f"forecast_steps={forecast_steps} exceeds max available aligned steps={max_steps}."
            )

        state = np.zeros(self.parameters.reservoir_size, dtype=np.float64)

        # Consume all warmup samples before the final teacher-forced input.
        # predict_one_step consumes x[warmup_steps - 1] exactly once and
        # predicts the value aligned with truth x[warmup_steps].
        for t in range(warmup_steps - 1):
            state = self._step(x_split[t], state)

        current = x_split[warmup_steps - 1].copy()
        preds = np.empty((forecast_steps, self.input_dim), dtype=np.float64)

        for k in range(forecast_steps):
            y_next, state = self.predict_one_step(current, state)

            unsafe = not np.all(np.isfinite(y_next))
            if max_abs is not None and not unsafe:
                with np.errstate(over="ignore", invalid="ignore"):
                    unsafe = bool(np.any(np.abs(y_next) > max_abs))

            if unsafe:
                preds[k:] = np.nan
                break

            preds[k] = y_next
            current = y_next

        return preds

    def to_metadata(self) -> dict[str, Any]:
        return {
            "parameters": asdict(self.parameters),
            "input_dim": self.input_dim,
            "feature_dimension": self.feature_dimension,
            "pre_scale_spectral_radius": self.pre_scale_spectral_radius,
            "reservoir_scale_factor": self.reservoir_scale_factor,
            "actual_spectral_radius": self.actual_spectral_radius,
            "is_fit": self.is_fit,
        }

    def save(self, path: str | Path) -> None:
        self._require_fit()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        assert self.w_res is not None
        assert self.win is not None
        assert self.w_out is not None

        np.savez_compressed(
            path,
            parameters_json=json.dumps(asdict(self.parameters), sort_keys=True),
            input_dim=np.asarray(self.input_dim, dtype=np.int64),
            feature_dimension=np.asarray(self.feature_dimension, dtype=np.int64),
            win=self.win,
            w_out=self.w_out,
            w_res_data=self.w_res.data,
            w_res_indices=self.w_res.indices,
            w_res_indptr=self.w_res.indptr,
            w_res_shape=np.asarray(self.w_res.shape, dtype=np.int64),
            pre_scale_spectral_radius=np.asarray(self.pre_scale_spectral_radius, dtype=np.float64),
            reservoir_scale_factor=np.asarray(self.reservoir_scale_factor, dtype=np.float64),
            actual_spectral_radius=np.asarray(self.actual_spectral_radius, dtype=np.float64),
        )

    @classmethod
    def load(cls, path: str | Path) -> _PERCReservoirBackbone:
        path = Path(path)

        with np.load(path, allow_pickle=False) as obj:
            parameters = PERCParameters(**json.loads(str(obj["parameters_json"])))
            model = cls(parameters)

            model.input_dim = int(obj["input_dim"])
            model.feature_dimension = int(obj["feature_dimension"])
            model.win = np.asarray(obj["win"], dtype=np.float64)
            model.w_out = np.asarray(obj["w_out"], dtype=np.float64)

            shape = tuple(int(v) for v in obj["w_res_shape"])
            model.w_res = sp.csr_matrix(
                (
                    np.asarray(obj["w_res_data"], dtype=np.float64),
                    np.asarray(obj["w_res_indices"], dtype=np.int64),
                    np.asarray(obj["w_res_indptr"], dtype=np.int64),
                ),
                shape=shape,
            )

            model.pre_scale_spectral_radius = float(obj["pre_scale_spectral_radius"])
            model.reservoir_scale_factor = float(obj["reservoir_scale_factor"])
            model.actual_spectral_radius = float(obj["actual_spectral_radius"])

        return model


# PERC-specific constrained readout extension.
Array = np.ndarray


class PERCModel(_PERCReservoirBackbone):
    """
    Physics-Enforced Reservoir Computing model.

    PERC uses the same deterministic ESN/RC reservoir backbone as RCModel, but
    replaces the unconstrained ridge readout with a hard equality-constrained
    readout.

    The prediction is

        y = y0 + phi(u, r) W,

    and the learned residual readout W is restricted so that every prediction
    satisfies

        C y = d.

    The runner must provide C and d in the same coordinate system where the
    model predicts. In our pipeline this will usually be standardized model
    coordinates, after transforming a physical-space constraint using train-only
    standardization statistics.
    """

    def __init__(self, parameters: PERCParameters):
        super().__init__(parameters)

        self.constraint_matrix: Array | None = None
        self.constraint_target: Array | None = None
        self.constraint_nullspace_basis: Array | None = None
        self.constraint_particular_solution: Array | None = None
        self.constraint_rank: int | None = None
        self.constraint_train_max_abs_violation: float | None = None
        self.constraint_train_mean_abs_violation: float | None = None

    @staticmethod
    def _as_2d_constraint_matrix(constraint_matrix: Array, *, output_dim: int) -> Array:
        matrix = np.asarray(constraint_matrix, dtype=np.float64)

        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        if matrix.ndim != 2:
            raise ValueError(f"constraint_matrix must be 1D or 2D, got shape {matrix.shape}.")

        if matrix.shape[1] != output_dim:
            raise ValueError(
                "constraint_matrix output dimension mismatch: "
                f"expected {output_dim}, got {matrix.shape[1]}."
            )

        if matrix.shape[0] <= 0:
            raise ValueError("constraint_matrix must contain at least one constraint row.")

        if not np.all(np.isfinite(matrix)):
            raise ValueError("constraint_matrix contains non-finite values.")

        if np.linalg.norm(matrix) == 0.0:
            raise ValueError("constraint_matrix must not be all zero.")

        return matrix

    @staticmethod
    def _as_1d_constraint_target(constraint_target: Array | float, *, n_constraints: int) -> Array:
        target = np.asarray(constraint_target, dtype=np.float64)

        if target.ndim == 0:
            target = target.reshape(1)

        if target.ndim != 1:
            raise ValueError(f"constraint_target must be scalar or 1D, got shape {target.shape}.")

        if target.shape[0] != n_constraints:
            raise ValueError(
                "constraint_target dimension mismatch: "
                f"expected {n_constraints}, got {target.shape[0]}."
            )

        if not np.all(np.isfinite(target)):
            raise ValueError("constraint_target contains non-finite values.")

        return target

    @staticmethod
    def _rank_tolerance(matrix: Array, *, rtol: float) -> float:
        singular_values = np.linalg.svd(matrix, compute_uv=False)

        if singular_values.size == 0:
            return float(rtol)

        return float(max(matrix.shape) * np.max(singular_values) * rtol)

    @classmethod
    def _constraint_nullspace(
        cls,
        matrix: Array,
        *,
        rtol: float,
    ) -> tuple[Array, int]:
        """
        Return an orthonormal basis for null(C).

        matrix shape: (n_constraints, output_dim)
        basis shape : (output_dim, nullity)
        """
        _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)

        tol = cls._rank_tolerance(matrix, rtol=rtol)
        rank = int(np.sum(singular_values > tol))

        basis = vh[rank:].T.astype(np.float64, copy=True)

        if basis.shape[0] != matrix.shape[1]:
            raise RuntimeError(
                "Internal nullspace basis output dimension mismatch: "
                f"basis={basis.shape}, matrix={matrix.shape}."
            )

        if basis.shape[1] <= 0:
            raise ValueError(
                "Constraint matrix leaves no free output dimension. "
                f"rank={rank}, output_dim={matrix.shape[1]}."
            )

        return basis, rank

    @staticmethod
    def _minimum_norm_particular_solution(matrix: Array, target: Array, *, rcond: float) -> Array:
        """
        Find y0 with C y0 = d using the minimum-norm row-space solution.
        """
        gram = matrix @ matrix.T
        y0 = matrix.T @ (np.linalg.pinv(gram, rcond=rcond) @ target)
        y0 = np.asarray(y0, dtype=np.float64)

        residual = matrix @ y0 - target
        max_abs_residual = float(np.max(np.abs(residual)))

        if max_abs_residual > 1e-8:
            raise ValueError(
                "Could not construct a particular solution satisfying C y = d. "
                f"max_abs_residual={max_abs_residual:.3e}."
            )

        return y0

    def fit_constrained(
        self,
        x_train: Array,
        *,
        constraint_matrix: Array,
        constraint_target: Array | float,
        constraint_rtol: float = 1e-10,
    ) -> None:
        """
        Fit a hard-constrained readout on a standardized/native training trajectory.

        x_train, constraint_matrix, and constraint_target must be expressed in
        the same coordinate system.
        """
        x_train = np.asarray(x_train, dtype=np.float64)

        if x_train.ndim != 2:
            raise ValueError(f"x_train must be 2D, got shape {x_train.shape}.")
        first_training_index = max(0, int(self.parameters.washout_steps) - 1)

        if x_train.shape[0] < first_training_index + 2:
            raise ValueError(
                "Training trajectory too short for washout and one-step targets: "
                f"length={x_train.shape[0]}, washout={self.parameters.washout_steps}, "
                f"first_training_index={first_training_index}."
            )

        if not self.parameters.include_bias:
            raise ValueError(
                "PERC requires include_bias=true so the affine constraint target d "
                "can be represented consistently."
            )

        if not self.is_initialized:
            self.initialize(input_dim=x_train.shape[1])

        if x_train.shape[1] != self.input_dim:
            raise ValueError(f"input_dim mismatch: model={self.input_dim}, train={x_train.shape[1]}.")

        constraint_matrix = self._as_2d_constraint_matrix(
            constraint_matrix,
            output_dim=x_train.shape[1],
        )
        constraint_target = self._as_1d_constraint_target(
            constraint_target,
            n_constraints=constraint_matrix.shape[0],
        )

        nullspace_basis, rank = self._constraint_nullspace(
            constraint_matrix,
            rtol=float(constraint_rtol),
        )

        particular_solution = self._minimum_norm_particular_solution(
            constraint_matrix,
            constraint_target,
            rcond=float(constraint_rtol),
        )

        state = np.zeros(self.parameters.reservoir_size, dtype=np.float64)

        features = []
        targets = []

        for t in range(x_train.shape[0] - 1):
            u_t = x_train[t]
            state = self._step(u_t, state)

            if t >= first_training_index:
                features.append(self._features(u_t, state))
                targets.append(x_train[t + 1])

        phi = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)

        if phi.ndim != 2 or y.ndim != 2:
            raise RuntimeError("Internal PERC training matrices have invalid shape.")

        residual_targets = y - particular_solution.reshape(1, -1)
        nullspace_targets = residual_targets @ nullspace_basis

        regressor = Ridge(
            alpha=self.parameters.ridge_alpha,
            fit_intercept=False,
            copy_X=False,
        )
        regressor.fit(phi, nullspace_targets)

        # sklearn stores coef_ as shape (nullity, feature_dim).
        w_null = regressor.coef_.T.astype(np.float64, copy=True)

        # Map free readout back to the original output coordinates.
        self.w_out = (w_null @ nullspace_basis.T).astype(np.float64, copy=True)

        self.constraint_matrix = constraint_matrix.astype(np.float64, copy=True)
        self.constraint_target = constraint_target.astype(np.float64, copy=True)
        self.constraint_nullspace_basis = nullspace_basis.astype(np.float64, copy=True)
        self.constraint_particular_solution = particular_solution.astype(np.float64, copy=True)
        self.constraint_rank = int(rank)

        train_pred = particular_solution.reshape(1, -1) + phi @ self.w_out
        violation = constraint_matrix @ train_pred.T - constraint_target.reshape(-1, 1)

        self.constraint_train_max_abs_violation = float(np.max(np.abs(violation)))
        self.constraint_train_mean_abs_violation = float(np.mean(np.abs(violation)))

    def predict_one_step(self, u: Array, state: Array) -> tuple[Array, Array]:
        self._require_fit()

        if self.constraint_particular_solution is None:
            raise RuntimeError("PERCModel is missing constraint_particular_solution.")

        with np.errstate(over="ignore", invalid="ignore"):
            state_next = self._step(u, state)
            phi = self._features(u, state_next)
            y_next = self.constraint_particular_solution + phi @ self.w_out

        return y_next.astype(np.float64, copy=False), state_next

    def to_metadata(self) -> dict[str, Any]:
        base = super().to_metadata()
        base.update(
            {
                "constraint_rank": self.constraint_rank,
                "constraint_matrix_shape": None
                if self.constraint_matrix is None
                else list(self.constraint_matrix.shape),
                "constraint_nullity": None
                if self.constraint_nullspace_basis is None
                else int(self.constraint_nullspace_basis.shape[1]),
                "constraint_target": None
                if self.constraint_target is None
                else self.constraint_target.tolist(),
                "constraint_train_max_abs_violation": self.constraint_train_max_abs_violation,
                "constraint_train_mean_abs_violation": self.constraint_train_mean_abs_violation,
            }
        )
        return base

    def save(self, path: str | Path) -> None:
        self._require_fit()

        if (
            self.constraint_matrix is None
            or self.constraint_target is None
            or self.constraint_nullspace_basis is None
            or self.constraint_particular_solution is None
        ):
            raise RuntimeError("PERCModel is missing fitted constraint metadata.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        assert self.w_res is not None
        assert self.win is not None
        assert self.w_out is not None

        np.savez_compressed(
            path,
            parameters_json=json.dumps(asdict(self.parameters), sort_keys=True),
            input_dim=np.asarray(self.input_dim, dtype=np.int64),
            feature_dimension=np.asarray(self.feature_dimension, dtype=np.int64),
            win=self.win,
            w_out=self.w_out,
            w_res_data=self.w_res.data,
            w_res_indices=self.w_res.indices,
            w_res_indptr=self.w_res.indptr,
            w_res_shape=np.asarray(self.w_res.shape, dtype=np.int64),
            pre_scale_spectral_radius=np.asarray(self.pre_scale_spectral_radius, dtype=np.float64),
            reservoir_scale_factor=np.asarray(self.reservoir_scale_factor, dtype=np.float64),
            actual_spectral_radius=np.asarray(self.actual_spectral_radius, dtype=np.float64),
            constraint_matrix=self.constraint_matrix,
            constraint_target=self.constraint_target,
            constraint_nullspace_basis=self.constraint_nullspace_basis,
            constraint_particular_solution=self.constraint_particular_solution,
            constraint_rank=np.asarray(self.constraint_rank, dtype=np.int64),
            constraint_train_max_abs_violation=np.asarray(
                self.constraint_train_max_abs_violation,
                dtype=np.float64,
            ),
            constraint_train_mean_abs_violation=np.asarray(
                self.constraint_train_mean_abs_violation,
                dtype=np.float64,
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> PERCModel:
        path = Path(path)

        with np.load(path, allow_pickle=False) as obj:
            parameters = PERCParameters(**json.loads(str(obj["parameters_json"])))
            model = cls(parameters)

            model.input_dim = int(obj["input_dim"])
            model.feature_dimension = int(obj["feature_dimension"])
            model.win = np.asarray(obj["win"], dtype=np.float64)
            model.w_out = np.asarray(obj["w_out"], dtype=np.float64)

            shape = tuple(int(v) for v in obj["w_res_shape"])
            model.w_res = sp.csr_matrix(
                (
                    np.asarray(obj["w_res_data"], dtype=np.float64),
                    np.asarray(obj["w_res_indices"], dtype=np.int64),
                    np.asarray(obj["w_res_indptr"], dtype=np.int64),
                ),
                shape=shape,
            )

            model.pre_scale_spectral_radius = float(obj["pre_scale_spectral_radius"])
            model.reservoir_scale_factor = float(obj["reservoir_scale_factor"])
            model.actual_spectral_radius = float(obj["actual_spectral_radius"])

            model.constraint_matrix = np.asarray(obj["constraint_matrix"], dtype=np.float64)
            model.constraint_target = np.asarray(obj["constraint_target"], dtype=np.float64)
            model.constraint_nullspace_basis = np.asarray(
                obj["constraint_nullspace_basis"],
                dtype=np.float64,
            )
            model.constraint_particular_solution = np.asarray(
                obj["constraint_particular_solution"],
                dtype=np.float64,
            )
            model.constraint_rank = int(obj["constraint_rank"])
            model.constraint_train_max_abs_violation = float(
                obj["constraint_train_max_abs_violation"]
            )
            model.constraint_train_mean_abs_violation = float(
                obj["constraint_train_mean_abs_violation"]
            )

        return model
