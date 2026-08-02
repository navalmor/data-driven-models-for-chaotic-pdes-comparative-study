from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forecasting.config import (
    ConfigDict,
    require_key,
    require_section,
    save_resolved_config_json,
    validate_experiment,
)
from forecasting.data import ForecastData, complete_trajectory_shape, load_forecast_data
from forecasting.decoding import decode_latent_file_to_physical_file
from forecasting.metrics import evaluate_forecast, save_error_curve_csv
from forecasting.models.perc import PERCModel, PERCParameters
from forecasting.preprocessing import Standardizer
from forecasting.registry import append_registry_row


logger = logging.getLogger(__name__)

Array = np.ndarray


@dataclass(frozen=True)
class PERCExperimentResult:
    run_id: str
    data_mode: str
    output_dir: Path
    summary_path: Path
    validation_relative_l2_horizon_time: float
    test_relative_l2_horizon_time: float | None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)
        return value if np.isfinite(value) else None

    if isinstance(value, float):
        return value if np.isfinite(value) else None

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    return value


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _require_bool(section: dict[str, Any], key: str, section_name: str) -> bool:
    value = require_key(section, key, section_name)

    if not isinstance(value, bool):
        raise TypeError(f"{section_name}.{key} must be boolean, got {value!r}.")

    return value


def _require_nonnegative_int(section: dict[str, Any], key: str, section_name: str) -> int:
    value = require_key(section, key, section_name)

    if isinstance(value, bool):
        raise TypeError(f"{section_name}.{key} must be integer, got bool.")

    value_int = int(value)

    if value_int != value:
        raise TypeError(f"{section_name}.{key} must be integer, got {value!r}.")

    if value_int < 0:
        raise ValueError(f"{section_name}.{key} must be non-negative, got {value_int}.")

    return value_int


def _require_positive_int(section: dict[str, Any], key: str, section_name: str) -> int:
    value = _require_nonnegative_int(section, key, section_name)

    if value <= 0:
        raise ValueError(f"{section_name}.{key} must be positive, got {value}.")

    return value


def _require_positive_float(section: dict[str, Any], key: str, section_name: str) -> float:
    value = require_key(section, key, section_name)

    if isinstance(value, bool):
        raise TypeError(f"{section_name}.{key} must be float, got bool.")

    value_float = float(value)

    if not np.isfinite(value_float) or value_float <= 0.0:
        raise ValueError(
            f"{section_name}.{key} must be positive and finite, got {value_float}."
        )

    return value_float


def _build_perc_parameters(cfg: ConfigDict) -> PERCParameters:
    model_cfg = require_section(cfg, "model")

    parameters = PERCParameters(
        reservoir_size=_require_positive_int(model_cfg, "reservoir_size", "model"),
        sparsity=_require_positive_float(model_cfg, "sparsity", "model"),
        spectral_radius=_require_positive_float(model_cfg, "spectral_radius", "model"),
        input_scaling=_require_positive_float(model_cfg, "input_scaling", "model"),
        leaking_rate=_require_positive_float(model_cfg, "leaking_rate", "model"),
        ridge_alpha=_require_positive_float(model_cfg, "ridge_alpha", "model"),
        washout_steps=_require_nonnegative_int(model_cfg, "washout_steps", "model"),
        include_bias=_require_bool(model_cfg, "include_bias", "model"),
        include_input_skip=_require_bool(model_cfg, "include_input_skip", "model"),
        random_seed=_require_nonnegative_int(model_cfg, "random_seed", "model"),
    )

    parameters.validate()
    return parameters


def _build_output_dir(
    *,
    cfg: ConfigDict,
    repo_root: Path,
    run_id: str,
) -> tuple[Path, bool]:
    experiment_cfg = require_section(cfg, "experiment")
    output_cfg = require_section(cfg, "output")

    model_name = str(require_key(experiment_cfg, "model", "experiment"))
    run_group = str(require_key(output_cfg, "run_group", "output")).strip("/")

    if not model_name:
        raise ValueError("experiment.model must be non-empty.")

    if not run_group:
        raise ValueError("output.run_group must be non-empty.")

    run_group_path = Path(run_group)

    if run_group_path.is_absolute() or any(part in {"..", ""} for part in run_group_path.parts):
        raise ValueError(f"output.run_group must be a safe relative path, got {run_group!r}.")

    base_dir = Path(require_key(output_cfg, "base_dir", "output")).expanduser()
    base_dir = base_dir if base_dir.is_absolute() else repo_root / base_dir

    overwrite = _require_bool(output_cfg, "overwrite", "output")

    output_dir = base_dir / model_name / run_group_path / run_id
    return output_dir.resolve(), overwrite


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists and overwrite=false: {output_dir}"
            )

        logger.warning("Overwrite enabled | removing=%s", output_dir)
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def _fit_standardizer_and_transform_splits(
    data: ForecastData,
    *,
    output_dir: Path,
) -> tuple[Standardizer, Array, Array, Array]:
    standardizer = Standardizer.fit(data.native_train)

    x_train = standardizer.transform(data.native_train, name="native_train")
    x_valid = standardizer.transform(data.native_valid, name="native_valid")
    x_test = standardizer.transform(data.native_test, name="native_test")

    standardizer.save(output_dir / "standardizer.npz")
    return standardizer, x_train, x_valid, x_test


def _first_unsafe_prediction_index(prediction: Array, *, max_abs: float) -> int | None:
    prediction = np.asarray(prediction, dtype=np.float64)

    if prediction.ndim != 2:
        raise ValueError(f"prediction must be 2D, got shape {prediction.shape}.")

    finite_rows = np.all(np.isfinite(prediction), axis=1)

    with np.errstate(over="ignore", invalid="ignore"):
        bounded_rows = np.all(np.abs(prediction) <= float(max_abs), axis=1)

    safe_rows = finite_rows & bounded_rows
    bad = np.flatnonzero(~safe_rows)

    return int(bad[0]) if bad.size else None


def _decode_latent_prediction_safely(
    *,
    repo_root: Path,
    data: ForecastData,
    pred_native: Array,
    output_dir: Path,
    split_name: str,
    prediction_safety_max_abs: float,
) -> tuple[Array, int | None, int]:
    if data.decoder_run_dir is None:
        raise ValueError("decoder_run_dir is required for latent decoding.")

    first_unsafe_index = _first_unsafe_prediction_index(
        pred_native,
        max_abs=prediction_safety_max_abs,
    )

    if first_unsafe_index is None:
        safe_prefix_length = int(pred_native.shape[0])
    else:
        safe_prefix_length = int(first_unsafe_index)
        logger.warning(
            "Unsafe latent rollout | split=%s first_unsafe=%d decoded_prefix=%d",
            split_name,
            first_unsafe_index,
            safe_prefix_length,
        )

    pred_physical = np.full(
        (pred_native.shape[0], data.physical_dimension),
        np.nan,
        dtype=np.float64,
    )

    if safe_prefix_length == 0:
        return pred_physical, first_unsafe_index, safe_prefix_length

    decode_dir = output_dir / "_tmp_decode"
    decode_dir.mkdir(parents=True, exist_ok=True)

    latent_prefix_path = decode_dir / f"{split_name}_latent_prefix.npy"
    physical_prefix_path = decode_dir / f"{split_name}_physical_prefix.npy"

    np.save(latent_prefix_path, pred_native[:safe_prefix_length].astype(np.float64))

    decode_result = decode_latent_file_to_physical_file(
        repo_root=repo_root,
        decoder_run_dir=data.decoder_run_dir,
        latent_input_path=latent_prefix_path,
        physical_output_path=physical_prefix_path,
        expected_latent_dim=data.native_dimension,
        expected_physical_dim=data.physical_dimension,
    )

    decoded_prefix = np.load(physical_prefix_path, allow_pickle=False).astype(np.float64)

    if decoded_prefix.shape != (safe_prefix_length, data.physical_dimension):
        raise RuntimeError(
            f"Decoded prefix shape mismatch for {split_name}: expected "
            f"{(safe_prefix_length, data.physical_dimension)}, got {decoded_prefix.shape}."
        )

    pred_physical[:safe_prefix_length] = decoded_prefix

    logger.debug(
        "Decoded latent prediction | split=%s latent_shape=%s physical_shape=%s",
        split_name,
        decode_result.latent_shape,
        decode_result.physical_shape,
    )

    return pred_physical, first_unsafe_index, safe_prefix_length


def _decode_latent_array_to_physical(
    *,
    repo_root: Path,
    data: ForecastData,
    latent_native: Array,
    output_dir: Path,
    label: str,
) -> Array:
    if data.decoder_run_dir is None:
        raise ValueError("decoder_run_dir is required for latent decoding.")

    latent_native = np.asarray(latent_native, dtype=np.float64)

    if latent_native.ndim != 2 or latent_native.shape[1] != int(data.native_dimension):
        raise ValueError(
            f"latent_native must have shape (n, {data.native_dimension}), "
            f"got {latent_native.shape}."
        )

    decode_dir = output_dir / "_tmp_decode"
    decode_dir.mkdir(parents=True, exist_ok=True)

    latent_path = decode_dir / f"{label}_latent.npy"
    physical_path = decode_dir / f"{label}_physical.npy"

    np.save(latent_path, latent_native)

    decode_result = decode_latent_file_to_physical_file(
        repo_root=repo_root,
        decoder_run_dir=data.decoder_run_dir,
        latent_input_path=latent_path,
        physical_output_path=physical_path,
        expected_latent_dim=data.native_dimension,
        expected_physical_dim=data.physical_dimension,
    )

    decoded = np.load(physical_path, allow_pickle=False).astype(np.float64)

    expected_shape = (latent_native.shape[0], data.physical_dimension)
    if decoded.shape != expected_shape:
        raise RuntimeError(
            f"Decoded latent array shape mismatch for {label}: expected "
            f"{expected_shape}, got {decoded.shape}."
        )

    logger.debug(
        "Decoded latent array | label=%s latent_shape=%s physical_shape=%s",
        label,
        decode_result.latent_shape,
        decode_result.physical_shape,
    )

    return decoded


def _prepare_physical_prediction(
    *,
    repo_root: Path,
    data: ForecastData,
    pred_native: Array,
    output_dir: Path,
    split_name: str,
    prediction_safety_max_abs: float,
) -> tuple[Array, bool, int | None, int]:
    if data.data_mode == "full64":
        first_unsafe = _first_unsafe_prediction_index(
            pred_native,
            max_abs=prediction_safety_max_abs,
        )
        safe_prefix_length = int(pred_native.shape[0]) if first_unsafe is None else int(first_unsafe)

        if first_unsafe is not None:
            logger.warning(
                "Unsafe physical rollout | split=%s first_unsafe=%d safe_prefix=%d",
                split_name,
                first_unsafe,
                safe_prefix_length,
            )

        return pred_native.astype(np.float64), False, first_unsafe, safe_prefix_length

    pred_physical, first_unsafe, safe_prefix_length = _decode_latent_prediction_safely(
        repo_root=repo_root,
        data=data,
        pred_native=pred_native,
        output_dir=output_dir,
        split_name=split_name,
        prediction_safety_max_abs=prediction_safety_max_abs,
    )

    return pred_physical, True, first_unsafe, safe_prefix_length




def _build_perc_integral_constraint(
    *,
    cfg: ConfigDict,
    data: ForecastData,
    standardizer: Standardizer,
    repo_root: Path,
    output_dir: Path,
) -> tuple[Array, Array, dict[str, Any], Array, float]:
    """Build a PERC integral constraint in standardized model space.

    Full64 uses the exact physical KS spatial-integral constraint because the
    model output is the physical field itself. Latent8 uses a train-only linear
    surrogate I(u) ~= a^T z + b because the true decoded constraint
    I(decoder(z)) is nonlinear in latent coordinates.
    """
    constraint_cfg = require_section(cfg, "constraint")

    constraint_type = str(require_key(constraint_cfg, "type", "constraint"))
    quadrature = str(constraint_cfg.get("quadrature", "rectangle"))
    domain_length = _require_positive_float(constraint_cfg, "domain_length", "constraint")
    target_mode = str(constraint_cfg.get("target_mode", "train_mean"))
    zero_tolerance = float(constraint_cfg.get("zero_tolerance", 1e-8))

    if zero_tolerance < 0.0 or not np.isfinite(zero_tolerance):
        raise ValueError(
            f"constraint.zero_tolerance must be non-negative finite, got {zero_tolerance}."
        )

    def _physical_integral_weights(n_physical: int) -> tuple[Array, float]:
        if n_physical <= 0:
            raise ValueError(f"physical dimension must be positive, got {n_physical}.")

        if quadrature == "rectangle":
            dx = float(domain_length) / float(n_physical)
            return np.full(n_physical, dx, dtype=np.float64), dx

        raise ValueError(
            "PERC currently supports constraint.quadrature='rectangle' only, "
            f"got {quadrature!r}."
        )

    def _target_from_training_integrals(
        train_integrals: Array,
    ) -> tuple[float, str, dict[str, Any]]:
        train_integrals = np.asarray(train_integrals, dtype=np.float64)

        if train_integrals.ndim != 1 or train_integrals.shape[0] == 0:
            raise ValueError(
                "training integrals must be a non-empty 1D array, "
                f"got shape {train_integrals.shape}."
            )

        if not np.all(np.isfinite(train_integrals)):
            raise ValueError("training integrals contain NaN or Inf.")

        train_integral_mean = float(np.mean(train_integrals))
        train_integral_std = float(np.std(train_integrals))
        train_integral_min = float(np.min(train_integrals))
        train_integral_max = float(np.max(train_integrals))
        train_integral_max_abs = float(np.max(np.abs(train_integrals)))

        if target_mode == "train_mean":
            target_integral = train_integral_mean
            target_reason = "training mean integral"
        elif target_mode == "zero":
            target_integral = 0.0
            target_reason = "configured zero target"
        elif target_mode == "zero_if_near_train_else_train_mean":
            if train_integral_max_abs <= zero_tolerance:
                target_integral = 0.0
                target_reason = "training integrals within zero_tolerance"
            else:
                target_integral = train_integral_mean
                target_reason = "training integral not near zero; using training mean"
        else:
            raise ValueError(
                "constraint.target_mode must be one of "
                "{'train_mean', 'zero', 'zero_if_near_train_else_train_mean'}, "
                f"got {target_mode!r}."
            )

        stats = {
            "train_integral_mean": train_integral_mean,
            "train_integral_std": train_integral_std,
            "train_integral_min": train_integral_min,
            "train_integral_max": train_integral_max,
            "train_integral_max_abs": train_integral_max_abs,
        }
        return float(target_integral), target_reason, stats

    def _finite_error_stats(errors: Array) -> dict[str, Any]:
        errors = np.asarray(errors, dtype=np.float64)
        finite = errors[np.isfinite(errors)]

        if finite.size == 0:
            return {
                "count": 0,
                "rmse": None,
                "mean_abs_error": None,
                "max_abs_error": None,
                "mean_error": None,
                "final_error": None,
            }

        return {
            "count": int(finite.size),
            "rmse": float(np.sqrt(np.mean(finite**2))),
            "mean_abs_error": float(np.mean(np.abs(finite))),
            "max_abs_error": float(np.max(np.abs(finite))),
            "mean_error": float(np.mean(finite)),
            "final_error": float(finite[-1]),
        }

    if constraint_type == "ks_spatial_integral":
        if data.data_mode != "full64":
            raise ValueError(
                "constraint.type='ks_spatial_integral' is exact full64-only. "
                "Use constraint.type='ks_spatial_integral_latent_linear_surrogate' "
                f"for latent8. Got data_mode={data.data_mode!r}."
            )

        n = int(data.native_dimension)
        if n != int(data.physical_dimension):
            raise ValueError(
                "PERC full64 constraint expects native_dimension == physical_dimension, "
                f"got native={data.native_dimension}, physical={data.physical_dimension}."
            )

        c_phys, dx = _physical_integral_weights(n)
        train_integrals = np.asarray(data.physical_train @ c_phys, dtype=np.float64)
        target_integral, target_reason, integral_stats = _target_from_training_integrals(
            train_integrals
        )

        mean = np.asarray(standardizer.mean, dtype=np.float64)
        scale = np.asarray(standardizer.scale, dtype=np.float64)

        if mean.shape != (n,) or scale.shape != (n,):
            raise ValueError(
                "Standardizer dimension mismatch for PERC full64 constraint: "
                f"mean={mean.shape}, scale={scale.shape}, n={n}."
            )

        c_std = c_phys * scale
        d_std = float(target_integral - float(c_phys @ mean))

        metadata = {
            "type": constraint_type,
            "constraint_enforced_space": "standardized_full64_model_space",
            "is_exact_physical_constraint": True,
            "quadrature": quadrature,
            "domain_length": float(domain_length),
            "dx": dx,
            "target_mode": target_mode,
            "target_reason": target_reason,
            "zero_tolerance": zero_tolerance,
            "target_integral_physical": float(target_integral),
            **integral_stats,
            "constraint_matrix_model_space_shape": [1, n],
            "constraint_matrix_model_space_norm": float(np.linalg.norm(c_std)),
            "constraint_target_model_space": [float(d_std)],
        }

        return (
            c_std.reshape(1, -1),
            np.asarray([d_std], dtype=np.float64),
            metadata,
            c_phys,
            float(target_integral),
        )

    if constraint_type != "ks_spatial_integral_latent_linear_surrogate":
        raise ValueError(
            "constraint.type must be 'ks_spatial_integral' or "
            "'ks_spatial_integral_latent_linear_surrogate', "
            f"got {constraint_type!r}."
        )

    if data.data_mode != "latent8":
        raise ValueError(
            "constraint.type='ks_spatial_integral_latent_linear_surrogate' "
            f"currently supports data_mode='latent8' only, got {data.data_mode!r}."
        )

    latent_dim = int(data.native_dimension)
    physical_dim = int(data.physical_dimension)

    c_phys, dx = _physical_integral_weights(physical_dim)

    true_train_integrals = np.asarray(data.physical_train @ c_phys, dtype=np.float64)
    target_integral, target_reason, integral_stats = _target_from_training_integrals(
        true_train_integrals
    )

    decoded_train_physical = _decode_latent_array_to_physical(
        repo_root=repo_root,
        data=data,
        latent_native=data.native_train,
        output_dir=output_dir,
        label="surrogate_train",
    )
    surrogate_train_integrals = np.asarray(decoded_train_physical @ c_phys, dtype=np.float64)

    surrogate_integral_stats = {
        "decoded_train_integral_mean": float(np.mean(surrogate_train_integrals)),
        "decoded_train_integral_std": float(np.std(surrogate_train_integrals)),
        "decoded_train_integral_min": float(np.min(surrogate_train_integrals)),
        "decoded_train_integral_max": float(np.max(surrogate_train_integrals)),
        "decoded_train_integral_max_abs": float(np.max(np.abs(surrogate_train_integrals))),
    }
    decoder_reconstruction_integral_error_stats = _finite_error_stats(
        surrogate_train_integrals - true_train_integrals
    )

    z_train = np.asarray(data.native_train, dtype=np.float64)
    if z_train.ndim != 2 or z_train.shape[1] != latent_dim:
        raise ValueError(
            "native_train must be a 2D latent array with expected dimension, "
            f"got shape={z_train.shape}, latent_dim={latent_dim}."
        )

    if z_train.shape[0] != surrogate_train_integrals.shape[0]:
        raise ValueError(
            "latent and physical training lengths differ for surrogate fit: "
            f"latent={z_train.shape[0]}, integrals={surrogate_train_integrals.shape[0]}."
        )

    surrogate_cfg = constraint_cfg.get("surrogate_fit", {})
    if surrogate_cfg is None:
        surrogate_cfg = {}
    if not isinstance(surrogate_cfg, dict):
        raise TypeError(
            "constraint.surrogate_fit must be an object/dict if provided, "
            f"got {type(surrogate_cfg).__name__}."
        )

    surrogate_method = str(surrogate_cfg.get("method", "ridge"))
    if surrogate_method != "ridge":
        raise ValueError(
            "constraint.surrogate_fit.method currently supports 'ridge' only, "
            f"got {surrogate_method!r}."
        )

    surrogate_alpha = float(surrogate_cfg.get("ridge_alpha", 1e-8))
    if surrogate_alpha < 0.0 or not np.isfinite(surrogate_alpha):
        raise ValueError(
            "constraint.surrogate_fit.ridge_alpha must be non-negative finite, "
            f"got {surrogate_alpha}."
        )

    include_intercept = surrogate_cfg.get("include_intercept", True)
    if not isinstance(include_intercept, bool):
        raise TypeError(
            "constraint.surrogate_fit.include_intercept must be boolean, "
            f"got {include_intercept!r}."
        )

    if include_intercept:
        design = np.column_stack(
            [z_train, np.ones(z_train.shape[0], dtype=np.float64)]
        )
        penalty = np.eye(latent_dim + 1, dtype=np.float64) * surrogate_alpha
        penalty[-1, -1] = 0.0
    else:
        design = z_train
        penalty = np.eye(latent_dim, dtype=np.float64) * surrogate_alpha

    lhs = design.T @ design + penalty
    rhs = design.T @ surrogate_train_integrals

    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs

    beta = np.asarray(beta, dtype=np.float64)
    if not np.all(np.isfinite(beta)):
        raise ValueError("latent integral surrogate fit produced non-finite coefficients.")

    if include_intercept:
        surrogate_slope = beta[:latent_dim]
        surrogate_intercept = float(beta[-1])
    else:
        surrogate_slope = beta
        surrogate_intercept = 0.0

    min_surrogate_slope_norm = float(
        surrogate_cfg.get("min_surrogate_slope_norm", 1e-12)
    )
    min_surrogate_response_std = float(
        surrogate_cfg.get("min_surrogate_response_std", 1e-10)
    )

    if min_surrogate_slope_norm < 0.0 or not np.isfinite(min_surrogate_slope_norm):
        raise ValueError(
            "constraint.surrogate_fit.min_surrogate_slope_norm must be non-negative finite, "
            f"got {min_surrogate_slope_norm}."
        )

    if min_surrogate_response_std < 0.0 or not np.isfinite(min_surrogate_response_std):
        raise ValueError(
            "constraint.surrogate_fit.min_surrogate_response_std must be non-negative finite, "
            f"got {min_surrogate_response_std}."
        )

    surrogate_slope_norm = float(np.linalg.norm(surrogate_slope))
    surrogate_response_std = float(np.std(surrogate_train_integrals))

    if surrogate_slope_norm <= min_surrogate_slope_norm:
        raise ValueError(
            "latent integral surrogate is degenerate: slope norm is too small. "
            f"slope_norm={surrogate_slope_norm}, "
            f"min_surrogate_slope_norm={min_surrogate_slope_norm}."
        )

    if surrogate_response_std <= min_surrogate_response_std:
        raise ValueError(
            "latent integral surrogate response is nearly constant and not useful. "
            f"decoded_train_integral_std={surrogate_response_std}, "
            f"min_surrogate_response_std={min_surrogate_response_std}."
        )

    train_surrogate_pred = z_train @ surrogate_slope + surrogate_intercept
    train_surrogate_errors = train_surrogate_pred - surrogate_train_integrals
    train_surrogate_stats = _finite_error_stats(train_surrogate_errors)

    mean = np.asarray(standardizer.mean, dtype=np.float64)
    scale = np.asarray(standardizer.scale, dtype=np.float64)

    if mean.shape != (latent_dim,) or scale.shape != (latent_dim,):
        raise ValueError(
            "Standardizer dimension mismatch for PERC latent surrogate: "
            f"mean={mean.shape}, scale={scale.shape}, latent_dim={latent_dim}."
        )

    c_std = surrogate_slope * scale
    d_std = float(
        target_integral
        - surrogate_intercept
        - float(surrogate_slope @ mean)
    )

    if float(np.linalg.norm(c_std)) == 0.0:
        raise ValueError(
            "standardized latent surrogate constraint vector is all zero; "
            "cannot build PERC constraint."
        )

    metadata = {
        "type": constraint_type,
        "constraint_enforced_space": "standardized_latent_model_space",
        "is_exact_physical_constraint": False,
        "physical_constraint_evaluated_after_decoding": True,
        "surrogate_statement": (
            "Train-only linear surrogate of physical spatial integral in latent space; "
            "decoded physical constraint error must be reported separately."
        ),
        "quadrature": quadrature,
        "domain_length": float(domain_length),
        "dx": dx,
        "target_mode": target_mode,
        "target_reason": target_reason,
        "zero_tolerance": zero_tolerance,
        "target_integral_physical": float(target_integral),
        **integral_stats,
        "surrogate_target": "decoded_training_latent_integral",
        "true_physical_target_source": "training_physical_integral_mean",
        "surrogate_integral_stats": surrogate_integral_stats,
        "decoder_reconstruction_integral_error": decoder_reconstruction_integral_error_stats,
        "surrogate_fit": {
            "method": surrogate_method,
            "ridge_alpha": surrogate_alpha,
            "include_intercept": include_intercept,
            "fit_on": "training_native_latent_and_decoded_training_latent_integral_only",
            "latent_dimension": latent_dim,
            "physical_dimension": physical_dim,
            "slope": surrogate_slope.tolist(),
            "intercept": surrogate_intercept,
            "slope_norm": surrogate_slope_norm,
            "decoded_train_integral_std": surrogate_response_std,
            "min_surrogate_slope_norm": min_surrogate_slope_norm,
            "min_surrogate_response_std": min_surrogate_response_std,
            "train_error": train_surrogate_stats,
        },
        "constraint_matrix_model_space_shape": [1, latent_dim],
        "constraint_matrix_model_space_norm": float(np.linalg.norm(c_std)),
        "constraint_target_model_space": [float(d_std)],
    }

    return (
        c_std.reshape(1, -1),
        np.asarray([d_std], dtype=np.float64),
        metadata,
        c_phys,
        float(target_integral),
    )


def _constraint_stats_physical(
    prediction_physical: Array,
    *,
    physical_weights: Array,
    target_integral: float,
    skip_initial: int,
) -> dict[str, Any]:
    prediction_physical = np.asarray(prediction_physical, dtype=np.float64)
    physical_weights = np.asarray(physical_weights, dtype=np.float64)

    scored = prediction_physical[int(skip_initial):]
    finite_rows = np.all(np.isfinite(scored), axis=1)
    finite_scored = scored[finite_rows]

    if finite_scored.shape[0] == 0:
        return {
            "finite_scored_steps": 0,
            "max_abs_error": None,
            "mean_abs_error": None,
            "final_error": None,
        }

    integrals = finite_scored @ physical_weights
    errors = integrals - float(target_integral)

    return {
        "finite_scored_steps": int(finite_scored.shape[0]),
        "max_abs_error": float(np.max(np.abs(errors))),
        "mean_abs_error": float(np.mean(np.abs(errors))),
        "final_error": float(errors[-1]),
    }


def _rollout_split(
    *,
    model: PERCModel,
    standardizer: Standardizer,
    split_scaled: Array,
    split_name: str,
    prediction_safety_max_abs: float,
) -> Array:
    """
    Teacher-force warmup samples, then autonomous PERC rollout.

    The returned prediction has the same length as split_scaled. The first
    warmup rows are copied from the true standardized split and excluded from
    metric evaluation via skip_initial=model.required_history_length. This
    matches the NGRC output contract while preserving RC one-step alignment:
        first generated prediction aligns with truth x[warmup].
    """
    warmup = int(model.required_history_length)

    if split_scaled.shape[0] <= warmup:
        raise ValueError(
            f"{split_name} split has {split_scaled.shape[0]} timesteps, "
            f"but warmup={warmup} requires at least warmup+1."
        )

    generated_scaled = model.rollout_from_split(
        split_scaled,
        warmup_steps=warmup,
        forecast_steps=int(split_scaled.shape[0] - warmup),
        prediction_safety_max_abs=prediction_safety_max_abs,
    )

    pred_scaled = np.empty_like(split_scaled, dtype=np.float64)
    pred_scaled[:warmup] = split_scaled[:warmup]
    pred_scaled[warmup:] = generated_scaled

    pred_native = standardizer.inverse_transform(
        pred_scaled,
        name=f"{split_name}_pred_scaled",
    )

    return pred_native.astype(np.float64)


def _save_split_outputs(
    *,
    split_dir: Path,
    truth_native: Array,
    pred_native: Array,
    truth_physical: Array,
    pred_physical: Array,
    metrics_physical: dict[str, Any],
    errors_physical: Any,
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)

    np.save(split_dir / "truth_native.npy", truth_native.astype(np.float64))
    np.save(split_dir / "pred_native.npy", pred_native.astype(np.float64))
    np.save(split_dir / "truth_physical.npy", truth_physical.astype(np.float64))
    np.save(split_dir / "pred_physical.npy", pred_physical.astype(np.float64))

    _write_json(split_dir / "metrics_physical.json", metrics_physical)
    save_error_curve_csv(split_dir / "errors_physical.csv", errors_physical)


def _format_shape_triplet(train: Array, valid: Array, test: Array) -> str:
    return f"{tuple(train.shape)}/{tuple(valid.shape)}/{tuple(test.shape)}"


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def run_perc_experiment(
    cfg: ConfigDict,
    *,
    repo_root: str | Path,
    config_path: str | Path | None = None,
    append_registry: bool = True,
    evaluate_test: bool = True,
) -> PERCExperimentResult:
    repo_root = Path(repo_root).expanduser().resolve()
    config_path_resolved = (
        Path(config_path).expanduser().resolve() if config_path is not None else None
    )

    validate_experiment(cfg, expected_type="single", expected_model="perc")

    if append_registry and not evaluate_test:
        raise ValueError(
            "append_registry=True requires evaluate_test=True because registry rows include test metrics."
        )

    experiment_cfg = require_section(cfg, "experiment")
    preprocessing_cfg = require_section(cfg, "preprocessing")
    evaluation_cfg = require_section(cfg, "evaluation")

    run_id = str(require_key(experiment_cfg, "run_id", "experiment"))
    data_mode = str(require_key(experiment_cfg, "data_mode", "experiment"))
    run_type = str(require_key(experiment_cfg, "type", "experiment"))

    if not run_id:
        raise ValueError("experiment.run_id must be non-empty.")

    output_dir, overwrite = _build_output_dir(
        cfg=cfg,
        repo_root=repo_root,
        run_id=run_id,
    )

    logger.info("PERC start | run=%s mode=%s", run_id, data_mode)

    _prepare_output_dir(output_dir, overwrite=overwrite)
    _write_json(output_dir / "config_input.json", cfg)

    data = load_forecast_data(cfg, repo_root=repo_root)

    logger.info(
        "Input trajectory: shape=%s, representation=%s, dtype=%s",
        complete_trajectory_shape(data.native_train, data.native_valid, data.native_test),
        "latent space" if "latent" in data_mode else "full state",
        data.native_train.dtype,
    )
    logger.debug(
        "Data splits: train=%s, validation=%s, test=%s",
        tuple(data.native_train.shape),
        tuple(data.native_valid.shape),
        tuple(data.native_test.shape),
    )
    logger.debug(
        "Data ready | native=%s physical=%s",
        _format_shape_triplet(data.native_train, data.native_valid, data.native_test),
        _format_shape_triplet(data.physical_train, data.physical_valid, data.physical_test),
    )

    standardize = _require_bool(preprocessing_cfg, "standardize", "preprocessing")
    if not standardize:
        raise ValueError("Final PERC pipeline requires preprocessing.standardize=true.")

    standardizer, x_train, x_valid, x_test = _fit_standardizer_and_transform_splits(
        data,
        output_dir=output_dir,
    )

    parameters = _build_perc_parameters(cfg)

    constraint_matrix, constraint_target, constraint_metadata, constraint_physical_weights, constraint_physical_target = (
        _build_perc_integral_constraint(
            cfg=cfg,
            data=data,
            standardizer=standardizer,
            repo_root=repo_root,
            output_dir=output_dir,
        )
    )

    model = PERCModel(parameters)
    model.fit_constrained(
        x_train,
        constraint_matrix=constraint_matrix,
        constraint_target=constraint_target,
    )

    model_path = output_dir / "perc_model.npz"
    model.save(model_path)

    logger.debug(
        "Model ready | reservoir=%d sparsity=%.4g sr=%.4g input_scale=%.4g "
        "leak=%.4g washout=%d features=%d ridge=%.3g bias=%s input_skip=%s",
        parameters.reservoir_size,
        parameters.sparsity,
        parameters.spectral_radius,
        parameters.input_scaling,
        parameters.leaking_rate,
        parameters.washout_steps,
        int(model.feature_dimension),
        parameters.ridge_alpha,
        str(parameters.include_bias).lower(),
        str(parameters.include_input_skip).lower(),
    )

    dt = _require_positive_float(evaluation_cfg, "dt", "evaluation")
    relative_l2_threshold = _require_positive_float(
        evaluation_cfg,
        "relative_l2_threshold",
        "evaluation",
    )
    prediction_safety_max_abs = _require_positive_float(
        evaluation_cfg,
        "prediction_safety_max_abs",
        "evaluation",
    )

    skip_initial = int(model.required_history_length)

    pred_valid_native = _rollout_split(
        model=model,
        standardizer=standardizer,
        split_scaled=x_valid,
        split_name="validation",
        prediction_safety_max_abs=prediction_safety_max_abs,
    )

    pred_valid_physical, decoded_valid, first_unsafe_valid, safe_valid_prefix = (
        _prepare_physical_prediction(
            repo_root=repo_root,
            data=data,
            pred_native=pred_valid_native,
            output_dir=output_dir,
            split_name="validation",
            prediction_safety_max_abs=prediction_safety_max_abs,
        )
    )

    valid_metrics_physical, valid_errors_physical = evaluate_forecast(
        data.physical_valid,
        pred_valid_physical,
        dt=dt,
        relative_l2_threshold=relative_l2_threshold,
        skip_initial=skip_initial,
        prediction_safety_max_abs=prediction_safety_max_abs,
    )

    _save_split_outputs(
        split_dir=output_dir / "validation",
        truth_native=data.native_valid,
        pred_native=pred_valid_native,
        truth_physical=data.physical_valid,
        pred_physical=pred_valid_physical,
        metrics_physical=valid_metrics_physical,
        errors_physical=valid_errors_physical,
    )

    valid_constraint_physical = _constraint_stats_physical(
        pred_valid_physical,
        physical_weights=constraint_physical_weights,
        target_integral=constraint_physical_target,
        skip_initial=skip_initial,
    )

    test_metrics_physical = None
    test_constraint_physical = None
    decoded_test = False
    first_unsafe_test = None
    safe_test_prefix = None

    if evaluate_test:
        pred_test_native = _rollout_split(
            model=model,
            standardizer=standardizer,
            split_scaled=x_test,
            split_name="test",
            prediction_safety_max_abs=prediction_safety_max_abs,
        )

        pred_test_physical, decoded_test, first_unsafe_test, safe_test_prefix = (
            _prepare_physical_prediction(
                repo_root=repo_root,
                data=data,
                pred_native=pred_test_native,
                output_dir=output_dir,
                split_name="test",
                prediction_safety_max_abs=prediction_safety_max_abs,
            )
        )

        test_metrics_physical, test_errors_physical = evaluate_forecast(
            data.physical_test,
            pred_test_physical,
            dt=dt,
            relative_l2_threshold=relative_l2_threshold,
            skip_initial=skip_initial,
            prediction_safety_max_abs=prediction_safety_max_abs,
        )

        _save_split_outputs(
            split_dir=output_dir / "test",
            truth_native=data.native_test,
            pred_native=pred_test_native,
            truth_physical=data.physical_test,
            pred_physical=pred_test_physical,
            metrics_physical=test_metrics_physical,
            errors_physical=test_errors_physical,
        )

        test_constraint_physical = _constraint_stats_physical(
            pred_test_physical,
            physical_weights=constraint_physical_weights,
            target_integral=constraint_physical_target,
            skip_initial=skip_initial,
        )

    decoded_to_physical = bool(decoded_valid or decoded_test)
    standardizer_path = output_dir / "standardizer.npz"

    resolved_config = {
        **cfg,
        "resolved": {
            "repo_root": repo_root,
            "config_path": config_path_resolved,
            "output_dir": output_dir,
            "model_path": model_path,
            "standardizer_path": standardizer_path,
        },
    }

    save_resolved_config_json(output_dir / "config_resolved.json", resolved_config)

    summary = {
        "run_id": run_id,
        "model_name": "perc",
        "run_type": run_type,
        "data_mode": data.data_mode,
        "native_dimension": data.native_dimension,
        "physical_dimension": data.physical_dimension,
        "train_end_index": data.train_end_index,
        "valid_end_index": data.valid_end_index,
        "standardized": True,
        "evaluate_test": bool(evaluate_test),
        "decoded_to_physical": decoded_to_physical,
        "selection_metric": "relative_l2_horizon",
        "valid_relative_l2_horizon_time": valid_metrics_physical["relative_l2_horizon_time"],
        "valid_relative_l2_mean": valid_metrics_physical["relative_l2_mean"],
        "valid_relative_l2_final": valid_metrics_physical["relative_l2_final"],
        "test_relative_l2_horizon_time": None
        if test_metrics_physical is None
        else test_metrics_physical["relative_l2_horizon_time"],
        "test_relative_l2_mean": None
        if test_metrics_physical is None
        else test_metrics_physical["relative_l2_mean"],
        "test_relative_l2_final": None
        if test_metrics_physical is None
        else test_metrics_physical["relative_l2_final"],
        "validation_constraint_physical_max_abs_error": valid_constraint_physical["max_abs_error"],
        "test_constraint_physical_max_abs_error": None
        if test_constraint_physical is None
        else test_constraint_physical["max_abs_error"],
        "perc_parameters": asdict(parameters),
        "required_history_length": model.required_history_length,
        "feature_dimension": int(model.feature_dimension),
        "pre_scale_spectral_radius": model.pre_scale_spectral_radius,
        "reservoir_scale_factor": model.reservoir_scale_factor,
        "actual_spectral_radius": model.actual_spectral_radius,
        "perc_constraint": constraint_metadata,
        "constraint_train_max_abs_violation_model_space": model.constraint_train_max_abs_violation,
        "constraint_train_mean_abs_violation_model_space": model.constraint_train_mean_abs_violation,
        "validation_constraint_physical": valid_constraint_physical,
        "test_constraint_physical": test_constraint_physical,
        "prediction_safety_max_abs": prediction_safety_max_abs,
        "first_unsafe_valid_native_index": first_unsafe_valid,
        "first_unsafe_test_native_index": first_unsafe_test,
        "safe_valid_prefix_length": safe_valid_prefix,
        "safe_test_prefix_length": safe_test_prefix,
        "validation_metrics_physical": valid_metrics_physical,
        "test_metrics_physical": test_metrics_physical,
        "paths": {
            "config_path": config_path_resolved,
            "output_dir": output_dir,
            "model_path": model_path,
            "standardizer_path": standardizer_path,
            "physical_data_path": data.physical_data_path,
            "native_data_dir": data.native_data_dir,
            "native_data_source": data.native_data_source,
            "decoder_run_dir": data.decoder_run_dir,
        },
    }

    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)

    registry_path = (
        repo_root
        / "experiments"
        / "forecasting"
        / "summary_tables"
        / "forecasting_experiment_registry.csv"
    )

    if append_registry:
        if test_metrics_physical is None:
            raise RuntimeError("Registry append requested without test metrics.")

        append_registry_row(
            registry_path,
            {
                "run_id": run_id,
                "model_name": "perc",
                "run_type": run_type,
                "data_mode": data.data_mode,
                "native_dimension": data.native_dimension,
                "physical_dimension": data.physical_dimension,
                "train_end_index": data.train_end_index,
                "valid_end_index": data.valid_end_index,
                "standardized": True,
                "decoded_to_physical": decoded_to_physical,
                "selection_metric": "relative_l2_horizon",
                "valid_relative_l2_horizon_time": valid_metrics_physical[
                    "relative_l2_horizon_time"
                ],
                "test_relative_l2_horizon_time": test_metrics_physical[
                    "relative_l2_horizon_time"
                ],
                "valid_relative_l2_mean": valid_metrics_physical["relative_l2_mean"],
                "test_relative_l2_mean": test_metrics_physical["relative_l2_mean"],
                "valid_relative_l2_final": valid_metrics_physical["relative_l2_final"],
                "test_relative_l2_final": test_metrics_physical["relative_l2_final"],
                "config_path": _relative_to_repo(config_path_resolved, repo_root)
                if config_path_resolved is not None
                else "",
                "output_dir": _relative_to_repo(output_dir, repo_root),
            },
        )

    valid_horizon = float(valid_metrics_physical["relative_l2_horizon_time"])
    test_horizon = (
        float(test_metrics_physical["relative_l2_horizon_time"])
        if test_metrics_physical is not None
        else None
    )

    if test_metrics_physical is None:
        logger.debug(
            "Result | valid_h=%.6g test_h=not_evaluated valid_mean=%s",
            valid_horizon,
            valid_metrics_physical["relative_l2_mean"],
        )
    else:
        logger.debug(
            "Result | valid_h=%.6g test_h=%.6g valid_mean=%s test_mean=%s",
            valid_horizon,
            test_horizon,
            valid_metrics_physical["relative_l2_mean"],
            test_metrics_physical["relative_l2_mean"],
        )

    # relative_l2_horizon_time is horizon_steps * dt (see forecasting.metrics), i.e.
    # physical simulation time -- the same quantity the public tables report as
    # validation_horizon_physical / test_horizon_physical. The step count is the
    # companion authoritative field; neither is recomputed here.
    logger.info("METRIC | Validation horizon (physical time): %s", valid_horizon)
    logger.debug(
        "Validation horizon: %s steps", valid_metrics_physical["relative_l2_horizon_steps"]
    )
    if test_horizon is not None:
        logger.info("METRIC | Test horizon (physical time): %s", test_horizon)
        logger.debug(
            "Test horizon: %s steps", test_metrics_physical["relative_l2_horizon_steps"]
        )
    logger.info("PERC done | summary=%s", _relative_to_repo(summary_path, repo_root))

    return PERCExperimentResult(
        run_id=run_id,
        data_mode=data.data_mode,
        output_dir=output_dir,
        summary_path=summary_path,
        validation_relative_l2_horizon_time=valid_horizon,
        test_relative_l2_horizon_time=test_horizon,
    )
