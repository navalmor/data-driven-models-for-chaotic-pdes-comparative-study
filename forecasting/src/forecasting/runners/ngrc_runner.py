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
from forecasting.models.ngrc import NGRCModel, NGRCParameters
from forecasting.preprocessing import Standardizer
from forecasting.registry import append_registry_row


logger = logging.getLogger(__name__)

Array = np.ndarray


@dataclass(frozen=True)
class NGRCExperimentResult:
    """
    Compact return object for one completed NGRC experiment.
    """

    run_id: str
    data_mode: str
    output_dir: Path
    summary_path: Path
    validation_relative_l2_horizon_time: float
    test_relative_l2_horizon_time: float | None


def _json_safe(value: Any) -> Any:
    """
    Convert runtime values into strict JSON-safe values.
    """
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
    """
    Save JSON with strict no-NaN policy.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    logger.debug("Saved JSON | path=%s", path)


def _require_bool(section: dict[str, Any], key: str, section_name: str) -> bool:
    value = require_key(section, key, section_name)

    if not isinstance(value, bool):
        raise TypeError(f"{section_name}.{key} must be a boolean, got {value!r}.")

    return value


def _require_positive_int(
    section: dict[str, Any],
    key: str,
    section_name: str,
) -> int:
    value = require_key(section, key, section_name)

    if isinstance(value, bool):
        raise TypeError(f"{section_name}.{key} must be an integer, got bool.")

    value_int = int(value)

    if value_int != value:
        raise TypeError(f"{section_name}.{key} must be an integer, got {value!r}.")

    if value_int <= 0:
        raise ValueError(f"{section_name}.{key} must be positive, got {value_int}.")

    return value_int


def _require_positive_float(
    section: dict[str, Any],
    key: str,
    section_name: str,
) -> float:
    value = require_key(section, key, section_name)

    if isinstance(value, bool):
        raise TypeError(f"{section_name}.{key} must be a float, got bool.")

    value_float = float(value)

    if not np.isfinite(value_float) or value_float <= 0.0:
        raise ValueError(
            f"{section_name}.{key} must be positive and finite, got {value_float}."
        )

    return value_float


def _build_ngrc_parameters(cfg: ConfigDict) -> NGRCParameters:
    """
    Build NGRCParameters from final JSON config.
    """
    model_cfg = require_section(cfg, "model")

    parameters = NGRCParameters(
        context_steps=_require_positive_int(model_cfg, "context_steps", "model"),
        delay_spacing=_require_positive_int(model_cfg, "delay_spacing", "model"),
        ridge_alpha=_require_positive_float(model_cfg, "ridge_alpha", "model"),
        include_linear=_require_bool(model_cfg, "include_linear", "model"),
        include_quadratic=_require_bool(model_cfg, "include_quadratic", "model"),
        quadratic_mode=str(require_key(model_cfg, "quadratic_mode", "model")),
        include_sin=_require_bool(model_cfg, "include_sin", "model"),
        include_cos=_require_bool(model_cfg, "include_cos", "model"),
        include_bias=_require_bool(model_cfg, "include_bias", "model"),
    )

    parameters.validate()
    return parameters


def _build_output_dir(
    *,
    cfg: ConfigDict,
    repo_root: Path,
    run_id: str,
) -> tuple[Path, bool]:
    """
    Resolve output directory and overwrite policy.

    Final thesis layout:
        experiments/forecasting/{model}/{run_group}/{run_id}

    Examples:
        experiments/forecasting/ngrc/selected/ngrc_full64_selected_v001
        experiments/forecasting/ngrc/search/grid/ngrc_full64_grid_search_v002
    """
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
        raise ValueError(
            f"output.run_group must be a safe relative path, got {run_group!r}."
        )

    base_dir = Path(require_key(output_cfg, "base_dir", "output")).expanduser()
    base_dir = base_dir if base_dir.is_absolute() else repo_root / base_dir

    overwrite = _require_bool(output_cfg, "overwrite", "output")

    output_dir = base_dir / model_name / run_group_path / run_id
    return output_dir.resolve(), overwrite


def _prepare_output_dir(
    output_dir: Path,
    *,
    overwrite: bool,
) -> None:
    """
    Create a clean output directory for the run.
    """
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
    """
    Fit standardizer on native training data only, then transform all splits.
    """
    standardizer = Standardizer.fit(data.native_train)

    x_train = standardizer.transform(data.native_train, name="native_train")
    x_valid = standardizer.transform(data.native_valid, name="native_valid")
    x_test = standardizer.transform(data.native_test, name="native_test")

    standardizer_path = output_dir / "standardizer.npz"
    standardizer.save(standardizer_path)

    return standardizer, x_train, x_valid, x_test


def _first_unsafe_prediction_index(
    prediction: Array,
    *,
    max_abs: float,
) -> int | None:
    """
    Return first row where prediction is non-finite or exceeds max_abs.
    """
    prediction = np.asarray(prediction, dtype=np.float64)

    if prediction.ndim != 2:
        raise ValueError(
            f"prediction must be 2D with shape (time_steps, dimension), "
            f"got {prediction.shape}."
        )

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
    """
    Decode latent prediction to physical 64D space.

    If the latent rollout becomes unsafe, only the safe prefix is decoded and
    the remaining physical prediction rows are filled with NaN.
    """
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


def _prepare_physical_prediction(
    *,
    repo_root: Path,
    data: ForecastData,
    pred_native: Array,
    output_dir: Path,
    split_name: str,
    prediction_safety_max_abs: float,
) -> tuple[Array, bool, int | None, int]:
    """
    Convert native prediction into physical-space prediction.
    """
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

        return (
            pred_native.astype(np.float64),
            False,
            first_unsafe,
            safe_prefix_length,
        )

    pred_physical, first_unsafe, safe_prefix_length = _decode_latent_prediction_safely(
        repo_root=repo_root,
        data=data,
        pred_native=pred_native,
        output_dir=output_dir,
        split_name=split_name,
        prediction_safety_max_abs=prediction_safety_max_abs,
    )

    return pred_physical, True, first_unsafe, safe_prefix_length


def _rollout_split(
    *,
    model: NGRCModel,
    standardizer: Standardizer,
    split_scaled: Array,
    split_name: str,
) -> Array:
    """
    Run autonomous NGRC rollout for one standardized native split.
    """
    required_history_length = model.required_history_length

    if split_scaled.shape[0] < required_history_length:
        raise ValueError(
            f"{split_name} split has {split_scaled.shape[0]} timesteps, "
            f"but required_history_length={required_history_length}."
        )

    pred_scaled = model.predict_autoregressive(
        initial_context=split_scaled[:required_history_length],
        n_steps=int(split_scaled.shape[0]),
    )

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
    """
    Save arrays, metrics, and error curve for one split.
    """
    split_dir.mkdir(parents=True, exist_ok=True)

    np.save(split_dir / "truth_native.npy", truth_native.astype(np.float64))
    np.save(split_dir / "pred_native.npy", pred_native.astype(np.float64))
    np.save(split_dir / "truth_physical.npy", truth_physical.astype(np.float64))
    np.save(split_dir / "pred_physical.npy", pred_physical.astype(np.float64))

    _write_json(split_dir / "metrics_physical.json", metrics_physical)
    save_error_curve_csv(split_dir / "errors_physical.csv", errors_physical)

    logger.debug("Saved split outputs | dir=%s", split_dir)


def _format_shape_triplet(train: Array, valid: Array, test: Array) -> str:
    """
    Compact shape formatter for terminal logs.
    """
    return f"{tuple(train.shape)}/{tuple(valid.shape)}/{tuple(test.shape)}"


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    """
    Return repo-relative path when possible.
    """
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def run_ngrc_experiment(
    cfg: ConfigDict,
    *,
    repo_root: str | Path,
    config_path: str | Path | None = None,
    append_registry: bool = True,
    evaluate_test: bool = True,
) -> NGRCExperimentResult:
    """
    Run one complete config-driven NGRC experiment.

    Workflow:
        1. validate config identity
        2. load full64/latent data
        3. fit train-only standardizer
        4. train NGRC on native training split
        5. roll out validation, and optionally test, autonomously
        6. decode latent predictions when needed
        7. evaluate in physical 64D space using relative-L2
        8. save outputs and optionally append registry row

    Search scripts should use evaluate_test=False so hyperparameter selection
    is validation-only. Final selected runs should use evaluate_test=True.
    """
    repo_root = Path(repo_root).expanduser().resolve()
    config_path_resolved = (
        Path(config_path).expanduser().resolve() if config_path is not None else None
    )

    validate_experiment(cfg, expected_type="single", expected_model="ngrc")

    if append_registry and not evaluate_test:
        raise ValueError(
            "append_registry=True requires evaluate_test=True because the final "
            "registry schema includes test metrics."
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

    logger.info(
        "NGRC start | run=%s mode=%s",
        run_id,
        data_mode,
    )

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
        _format_shape_triplet(
            data.physical_train,
            data.physical_valid,
            data.physical_test,
        ),
    )

    standardize = _require_bool(preprocessing_cfg, "standardize", "preprocessing")

    if not standardize:
        raise ValueError("Final NGRC pipeline requires preprocessing.standardize=true.")

    standardizer, x_train, x_valid, x_test = _fit_standardizer_and_transform_splits(
        data,
        output_dir=output_dir,
    )

    parameters = _build_ngrc_parameters(cfg)

    model = NGRCModel(parameters)
    model.fit(x_train)

    model_path = output_dir / "ngrc_model.npz"
    model.save(model_path)

    logger.debug(
        "Model ready | context=%d spacing=%d history=%d features=%d ridge=%.3g "
        "linear=%s quadratic=%s quad=%s sin=%s cos=%s bias=%s",
        parameters.context_steps,
        parameters.delay_spacing,
        model.required_history_length,
        model.feature_dimension,
        parameters.ridge_alpha,
        str(parameters.include_linear).lower(),
        str(parameters.include_quadratic).lower(),
        parameters.quadratic_mode,
        str(parameters.include_sin).lower(),
        str(parameters.include_cos).lower(),
        str(parameters.include_bias).lower(),
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

    pred_test_native = None
    pred_test_physical = None
    test_metrics_physical = None
    test_errors_physical = None
    decoded_test = False
    first_unsafe_test = None
    safe_test_prefix = None

    if evaluate_test:
        pred_test_native = _rollout_split(
            model=model,
            standardizer=standardizer,
            split_scaled=x_test,
            split_name="test",
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
        "model_name": "ngrc",
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
        "ngrc_parameters": asdict(parameters),
        "required_history_length": model.required_history_length,
        "feature_dimension": model.feature_dimension,
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
            raise RuntimeError("Internal error: registry append requested without test metrics.")

        append_registry_row(
            registry_path,
            {
                "run_id": run_id,
                "model_name": "ngrc",
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
    logger.info(
        "NGRC done | summary=%s",
        _relative_to_repo(summary_path, repo_root),
    )

    return NGRCExperimentResult(
        run_id=run_id,
        data_mode=data.data_mode,
        output_dir=output_dir,
        summary_path=summary_path,
        validation_relative_l2_horizon_time=valid_horizon,
        test_relative_l2_horizon_time=test_horizon,
    )
