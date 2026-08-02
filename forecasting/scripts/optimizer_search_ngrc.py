#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from skopt import Optimizer
from skopt.space import Categorical, Integer, Real


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
FORECASTING_SRC = REPO_ROOT / "forecasting" / "src"

for _bootstrap_path in (REPO_ROOT, FORECASTING_SRC):
    if str(_bootstrap_path) not in sys.path:
        sys.path.insert(0, str(_bootstrap_path))

from common.logging_setup import add_logging_arguments, configure_from_args, get_logger
from forecasting.config import ConfigDict, load_config, require_key, require_section
from forecasting.runners.ngrc_runner import run_ngrc_experiment


logger = get_logger("forecasting.scripts.optimizer_search_ngrc")

SUPPORTED_OPTIMIZERS = {"dummy", "forest", "gbrt", "gp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one JSON-configured NGRC optimizer search."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the final JSON NGRC optimizer-search config.",
    )
    add_logging_arguments(parser)
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)
        return value if np.isfinite(value) else None

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _require_list(section: dict[str, Any], key: str, section_name: str) -> list[Any]:
    value = require_key(section, key, section_name)

    if not isinstance(value, list) or not value:
        raise TypeError(f"{section_name}.{key} must be a non-empty list.")

    return value


def _require_positive_int(section: dict[str, Any], key: str, section_name: str) -> int:
    value = require_key(section, key, section_name)

    if isinstance(value, bool):
        raise TypeError(f"{section_name}.{key} must be an integer, got bool.")

    value_int = int(value)

    if value_int != value:
        raise TypeError(f"{section_name}.{key} must be an integer, got {value!r}.")

    if value_int <= 0:
        raise ValueError(f"{section_name}.{key} must be positive, got {value_int}.")

    return value_int


def _require_bool(section: dict[str, Any], key: str, section_name: str) -> bool:
    value = require_key(section, key, section_name)

    if not isinstance(value, bool):
        raise TypeError(f"{section_name}.{key} must be a boolean, got {value!r}.")

    return value


def _require_bool_list(section: dict[str, Any], key: str, section_name: str) -> list[bool]:
    values = _require_list(section, key, section_name)

    if not all(isinstance(value, bool) for value in values):
        raise TypeError(f"{section_name}.{key} must contain boolean values only.")

    return values


def _feature_flags(feature_set: str) -> dict[str, bool]:
    if feature_set == "linear":
        return {
            "include_linear": True,
            "include_quadratic": False,
            "include_sin": False,
            "include_cos": False,
        }

    if feature_set == "linear_quadratic":
        return {
            "include_linear": True,
            "include_quadratic": True,
            "include_sin": False,
            "include_cos": False,
        }

    if feature_set == "linear_trig":
        return {
            "include_linear": True,
            "include_quadratic": False,
            "include_sin": True,
            "include_cos": True,
        }

    if feature_set == "full":
        return {
            "include_linear": True,
            "include_quadratic": True,
            "include_sin": True,
            "include_cos": True,
        }

    raise ValueError(
        f"Unknown feature_set={feature_set!r}. "
        "Supported values: linear, linear_quadratic, linear_trig, full."
    )


def _safe_float_tag(value: float) -> str:
    return f"{float(value):.1e}".replace("+", "").replace("-", "m").replace(".", "p")


def _native_dimension_from_data_mode(data_mode: str) -> int:
    if data_mode == "full64":
        return 64
    if data_mode == "latent8":
        return 8
    if data_mode == "latent6":
        return 6

    raise ValueError(f"Unsupported data_mode={data_mode!r}.")


def _estimate_feature_dimension(
    *,
    native_dimension: int,
    context_steps: int,
    include_linear: bool,
    include_quadratic: bool,
    quadratic_mode: str,
    include_sin: bool,
    include_cos: bool,
    include_bias: bool,
) -> int:
    flat_dim = int(native_dimension) * int(context_steps)
    feature_dim = 0

    if include_bias:
        feature_dim += 1

    if include_linear:
        feature_dim += flat_dim

    if include_quadratic:
        if quadratic_mode == "diagonal":
            feature_dim += flat_dim
        elif quadratic_mode == "pairwise":
            feature_dim += flat_dim * (flat_dim + 1) // 2
        else:
            raise ValueError(f"Unknown quadratic_mode={quadratic_mode!r}.")

    if include_sin:
        feature_dim += flat_dim

    if include_cos:
        feature_dim += flat_dim

    return int(feature_dim)


def _candidate_run_id(
    *,
    search_run_id: str,
    optimizer_name: str,
    trial: int,
    feature_set: str,
    context_steps: int,
    delay_spacing: int,
    ridge_alpha: float,
    quadratic_mode: str,
    include_bias: bool,
) -> str:
    bias_tag = "bias" if include_bias else "nobias"

    return (
        f"{search_run_id}_{optimizer_name}_trial{trial:04d}_"
        f"{feature_set}_c{context_steps}_d{delay_spacing}_"
        f"{quadratic_mode}_{bias_tag}_ridge{_safe_float_tag(ridge_alpha)}"
    )


def _score_from_summary(summary: dict[str, Any]) -> tuple[float, float, float]:
    valid_metrics = summary["validation_metrics_physical"]

    horizon = float(valid_metrics["relative_l2_horizon_time"])
    finite_fraction = float(valid_metrics["finite_relative_l2_fraction"])

    mean_value = valid_metrics["relative_l2_mean"]
    mean_for_sort = float("inf") if mean_value is None else float(mean_value)

    return (-horizon, -finite_fraction, mean_for_sort)


def _objective_from_summary(summary: dict[str, Any]) -> float:
    """
    Scalar objective minimized by skopt.

    The main goal is maximizing validation horizon. The additional small terms
    help break ties in a stable ML-style way.
    """
    valid_metrics = summary["validation_metrics_physical"]

    horizon = float(valid_metrics["relative_l2_horizon_time"])
    finite_fraction = float(valid_metrics["finite_relative_l2_fraction"])

    mean_value = valid_metrics["relative_l2_mean"]
    mean_for_sort = 1e6 if mean_value is None else float(mean_value)

    return float(-horizon + (1.0 - finite_fraction) * 1000.0 + mean_for_sort * 1e-6)


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _display_path(path: Path) -> str:
    path = Path(path)

    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


OPTIMIZER_NAMES: tuple[str, ...] = ("gp", "dummy", "forest", "gbrt")


def _parse_optimizer_entries(search_cfg: dict[str, Any]) -> list[tuple[str, int]]:
    """
    Build (name, random_seed) pairs for the 4 required optimizer families.

    The optimizer families are fixed by OPTIMIZER_NAMES, not configurable,
    per the project rule that every optimizer search always covers
    dummy/gp/gbrt/forest together. Only each family's explicit seed comes
    from search.optimizer_seeds.
    """
    seeds_cfg = require_section(search_cfg, "optimizer_seeds")

    entries: list[tuple[str, int]] = []
    for name in OPTIMIZER_NAMES:
        seed = require_key(seeds_cfg, name, "search.optimizer_seeds")
        entries.append((name, int(seed)))

    return entries


def _validate_search_config(cfg: ConfigDict) -> None:
    experiment_cfg = require_section(cfg, "experiment")

    if require_key(experiment_cfg, "type", "experiment") != "optimizer_search":
        raise ValueError(
            "optimizer_search_ngrc.py requires experiment.type='optimizer_search'."
        )

    if require_key(experiment_cfg, "model", "experiment") != "ngrc":
        raise ValueError("optimizer_search_ngrc.py requires experiment.model='ngrc'.")


def _search_output_dir(cfg: ConfigDict) -> Path:
    experiment_cfg = require_section(cfg, "experiment")
    output_cfg = require_section(cfg, "output")

    model_name = str(require_key(experiment_cfg, "model", "experiment"))
    search_run_id = str(require_key(experiment_cfg, "run_id", "experiment"))
    run_group = str(require_key(output_cfg, "run_group", "output")).strip("/")

    base_dir = Path(require_key(output_cfg, "base_dir", "output")).expanduser()
    base_dir = base_dir if base_dir.is_absolute() else REPO_ROOT / base_dir

    return (base_dir / model_name / run_group / search_run_id).resolve()


def _make_dimensions(search_cfg: dict[str, Any]) -> list[Any]:
    return [
        Categorical(
            [str(x) for x in _require_list(search_cfg, "feature_sets", "search")],
            name="feature_set",
        ),
        Integer(
            int(require_key(search_cfg, "context_steps_min", "search")),
            int(require_key(search_cfg, "context_steps_max", "search")),
            name="context_steps",
        ),
        Categorical(
            [int(x) for x in _require_list(search_cfg, "delay_spacing_values", "search")],
            name="delay_spacing",
        ),
        Real(
            float(require_key(search_cfg, "ridge_alpha_min", "search")),
            float(require_key(search_cfg, "ridge_alpha_max", "search")),
            prior="log-uniform",
            name="ridge_alpha",
        ),
        Categorical(
            [str(x) for x in _require_list(search_cfg, "quadratic_modes", "search")],
            name="quadratic_mode",
        ),
        Categorical(
            _require_bool_list(search_cfg, "include_bias_values", "search"),
            name="include_bias",
        ),
    ]


def _random_sample(dimensions: list[Any], rng: np.random.Generator) -> list[Any]:
    values: list[Any] = []

    for dim in dimensions:
        if isinstance(dim, Categorical):
            values.append(rng.choice(list(dim.categories)).item())

        elif isinstance(dim, Integer):
            values.append(int(rng.integers(int(dim.low), int(dim.high) + 1)))

        elif isinstance(dim, Real):
            if dim.prior == "log-uniform":
                low = np.log10(float(dim.low))
                high = np.log10(float(dim.high))
                values.append(float(10.0 ** rng.uniform(low, high)))
            else:
                values.append(float(rng.uniform(float(dim.low), float(dim.high))))

        else:
            raise TypeError(f"Unsupported skopt dimension: {dim!r}")

    return values


def _optimizer_for_name(
    *,
    name: str,
    dimensions: list[Any],
    n_initial_points: int,
    random_state: int,
) -> Optimizer | None:
    if name == "dummy":
        return None

    estimator = {
        "gp": "GP",
        "forest": "RF",
        "gbrt": "GBRT",
    }[name]

    return Optimizer(
        dimensions=dimensions,
        base_estimator=estimator,
        n_initial_points=n_initial_points,
        random_state=random_state,
    )


def _params_to_dict(params: list[Any]) -> dict[str, Any]:
    return {
        "feature_set": str(params[0]),
        "context_steps": int(params[1]),
        "delay_spacing": int(params[2]),
        "ridge_alpha": float(params[3]),
        "quadratic_mode": str(params[4]),
        "include_bias": bool(params[5]),
    }


def _build_candidate_config(
    *,
    cfg: ConfigDict,
    optimizer_name: str,
    trial: int,
    params: dict[str, Any],
) -> tuple[ConfigDict, str, int]:
    experiment_cfg = require_section(cfg, "experiment")
    output_cfg = require_section(cfg, "output")

    search_run_id = str(require_key(experiment_cfg, "run_id", "experiment"))
    base_run_group = str(require_key(output_cfg, "run_group", "output")).strip("/")

    flags = _feature_flags(params["feature_set"])

    quadratic_mode = str(params["quadratic_mode"])
    if not flags["include_quadratic"]:
        quadratic_mode = "diagonal"

    candidate_run_id = _candidate_run_id(
        search_run_id=search_run_id,
        optimizer_name=optimizer_name,
        trial=trial,
        feature_set=params["feature_set"],
        context_steps=params["context_steps"],
        delay_spacing=params["delay_spacing"],
        ridge_alpha=params["ridge_alpha"],
        quadratic_mode=quadratic_mode,
        include_bias=params["include_bias"],
    )

    candidate_cfg = copy.deepcopy(cfg)
    candidate_cfg["experiment"]["run_id"] = candidate_run_id
    candidate_cfg["experiment"]["type"] = "single"
    candidate_cfg["output"]["run_group"] = (
        f"{base_run_group}/{search_run_id}/{optimizer_name}/trials"
    )

    candidate_cfg["model"].update(flags)
    candidate_cfg["model"]["context_steps"] = int(params["context_steps"])
    candidate_cfg["model"]["delay_spacing"] = int(params["delay_spacing"])
    candidate_cfg["model"]["ridge_alpha"] = float(params["ridge_alpha"])
    candidate_cfg["model"]["quadratic_mode"] = quadratic_mode
    candidate_cfg["model"]["include_bias"] = bool(params["include_bias"])

    candidate_cfg.pop("search", None)

    native_dimension = _native_dimension_from_data_mode(
        str(require_key(experiment_cfg, "data_mode", "experiment"))
    )

    feature_dimension = _estimate_feature_dimension(
        native_dimension=native_dimension,
        context_steps=int(params["context_steps"]),
        include_linear=bool(candidate_cfg["model"]["include_linear"]),
        include_quadratic=bool(candidate_cfg["model"]["include_quadratic"]),
        quadratic_mode=quadratic_mode,
        include_sin=bool(candidate_cfg["model"]["include_sin"]),
        include_cos=bool(candidate_cfg["model"]["include_cos"]),
        include_bias=bool(candidate_cfg["model"]["include_bias"]),
    )

    return candidate_cfg, candidate_run_id, feature_dimension


def _run_one_optimizer(
    *,
    cfg: ConfigDict,
    optimizer_name: str,
    dimensions: list[Any],
    n_calls: int,
    n_initial_points: int,
    random_state: int,
    search_out_dir: Path,
    max_feature_dimension: int,
) -> dict[str, Any]:
    if optimizer_name not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"Unsupported optimizer={optimizer_name!r}. "
            f"Supported values are {sorted(SUPPORTED_OPTIMIZERS)}."
        )

    optimizer_dir = search_out_dir / optimizer_name
    optimizer_dir.mkdir(parents=True, exist_ok=True)

    results_path = optimizer_dir / "optimizer_results.csv"

    fieldnames = [
        "optimizer",
        "trial",
        "candidate_run_id",
        "feature_set",
        "context_steps",
        "delay_spacing",
        "ridge_alpha",
        "quadratic_mode",
        "include_bias",
        "feature_dimension",
        "status",
        "objective_value",
        "valid_relative_l2_horizon_time",
        "valid_relative_l2_mean",
        "valid_relative_l2_final",
        "finite_relative_l2_fraction",
        "output_dir",
        "error",
    ]

    skopt_optimizer = _optimizer_for_name(
        name=optimizer_name,
        dimensions=dimensions,
        n_initial_points=n_initial_points,
        random_state=random_state,
    )

    rng = np.random.default_rng(random_state)

    rows: list[dict[str, Any]] = []

    runner_logger = logging.getLogger("forecasting.runners.ngrc_runner")
    old_runner_level = runner_logger.level
    runner_logger.setLevel(logging.WARNING)

    try:
        with results_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for trial in range(1, n_calls + 1):
                if skopt_optimizer is None:
                    raw_params = _random_sample(dimensions, rng)
                else:
                    raw_params = skopt_optimizer.ask()

                params = _params_to_dict(raw_params)

                candidate_cfg, candidate_run_id, feature_dimension = _build_candidate_config(
                    cfg=cfg,
                    optimizer_name=optimizer_name,
                    trial=trial,
                    params=params,
                )

                row_base = {
                    "optimizer": optimizer_name,
                    "trial": trial,
                    "candidate_run_id": candidate_run_id,
                    "feature_set": params["feature_set"],
                    "context_steps": params["context_steps"],
                    "delay_spacing": params["delay_spacing"],
                    "ridge_alpha": params["ridge_alpha"],
                    "quadratic_mode": candidate_cfg["model"]["quadratic_mode"],
                    "include_bias": params["include_bias"],
                    "feature_dimension": feature_dimension,
                }

                logger.info(
                    "Trial start | optimizer=%s trial=%04d/%04d run=%s features=%d",
                    optimizer_name,
                    trial,
                    n_calls,
                    candidate_run_id,
                    feature_dimension,
                )

                if feature_dimension > max_feature_dimension:
                    objective_value = 1e9
                    row = {
                        **row_base,
                        "status": "skipped_feature_dimension",
                        "objective_value": objective_value,
                        "valid_relative_l2_horizon_time": -1.0,
                        "valid_relative_l2_mean": "",
                        "valid_relative_l2_final": "",
                        "finite_relative_l2_fraction": 0.0,
                        "output_dir": "",
                        "error": (
                            f"feature_dimension={feature_dimension} exceeds "
                            f"max_feature_dimension={max_feature_dimension}"
                        ),
                        "_summary": None,
                        "_config": candidate_cfg,
                    }

                    logger.warning(
                        "Trial skipped | run=%s feature_dimension=%d max=%d",
                        candidate_run_id,
                        feature_dimension,
                        max_feature_dimension,
                    )

                else:
                    try:
                        result = run_ngrc_experiment(
                            candidate_cfg,
                            repo_root=REPO_ROOT,
                            config_path=None,
                            append_registry=False,
                            evaluate_test=False,
                        )

                        summary = _load_summary(result.summary_path)
                        valid_metrics = summary["validation_metrics_physical"]
                        objective_value = _objective_from_summary(summary)

                        row = {
                            **row_base,
                            "status": "ok",
                            "objective_value": objective_value,
                            "valid_relative_l2_horizon_time": valid_metrics[
                                "relative_l2_horizon_time"
                            ],
                            "valid_relative_l2_mean": valid_metrics["relative_l2_mean"],
                            "valid_relative_l2_final": valid_metrics["relative_l2_final"],
                            "finite_relative_l2_fraction": valid_metrics[
                                "finite_relative_l2_fraction"
                            ],
                            "output_dir": _display_path(result.output_dir),
                            "error": "",
                            "_summary": summary,
                            "_config": candidate_cfg,
                        }

                        logger.info(
                            "Trial done | optimizer=%s run=%s valid_h=%.6g mean=%s",
                            optimizer_name,
                            candidate_run_id,
                            valid_metrics["relative_l2_horizon_time"],
                            valid_metrics["relative_l2_mean"],
                        )

                    except Exception as exc:
                        objective_value = 1e9
                        row = {
                            **row_base,
                            "status": "failed",
                            "objective_value": objective_value,
                            "valid_relative_l2_horizon_time": -1.0,
                            "valid_relative_l2_mean": "",
                            "valid_relative_l2_final": "",
                            "finite_relative_l2_fraction": 0.0,
                            "output_dir": "",
                            "error": repr(exc),
                            "_summary": None,
                            "_config": candidate_cfg,
                        }

                        logger.exception("Trial failed | optimizer=%s run=%s", optimizer_name, candidate_run_id)

                if skopt_optimizer is not None:
                    skopt_optimizer.tell(raw_params, float(row["objective_value"]))

                rows.append(row)
                writer.writerow({key: row.get(key, "") for key in fieldnames})
                f.flush()

    finally:
        runner_logger.setLevel(old_runner_level)

    successful = [row for row in rows if row["status"] == "ok"]

    if not successful:
        raise RuntimeError(f"All trials failed for optimizer={optimizer_name}.")

    best_row = sorted(successful, key=lambda row: _score_from_summary(row["_summary"]))[0]

    best_cfg = copy.deepcopy(best_row["_config"])
    _write_json(optimizer_dir / "best_config_validation.json", best_cfg)
    _write_json(optimizer_dir / "best_summary.json", best_row["_summary"])

    return best_row


def main() -> int:
    args = _parse_args()

    try:
        cfg = load_config(args.config)

        logging_cfg = require_section(cfg, "logging")
        configure_from_args(args, component="OPT-NGRC",
                            config_level=str(require_key(logging_cfg, "level", "logging")))

        _validate_search_config(cfg)

        search_cfg = require_section(cfg, "search")
        output_cfg = require_section(cfg, "output")
        experiment_cfg = require_section(cfg, "experiment")

        optimizer_entries = _parse_optimizer_entries(search_cfg)
        n_calls = _require_positive_int(search_cfg, "n_calls", "search")
        n_initial_points = _require_positive_int(search_cfg, "n_initial_points", "search")

        if n_initial_points > n_calls:
            raise ValueError(
                "search.n_initial_points must be less than or equal to search.n_calls."
            )

        max_feature_dimension = _require_positive_int(
            search_cfg,
            "max_feature_dimension",
            "search",
        )
        append_best_to_registry = _require_bool(
            search_cfg,
            "append_best_to_registry",
            "search",
        )

        search_out_dir = _search_output_dir(cfg)
        overwrite = _require_bool(output_cfg, "overwrite", "output")

        if search_out_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Search output directory exists and overwrite=false: {search_out_dir}"
                )

            logger.warning("Overwrite enabled | removing=%s", search_out_dir)
            shutil.rmtree(search_out_dir)

        search_out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(search_out_dir / "search_config_input.json", cfg)

        dimensions = _make_dimensions(search_cfg)

        logger.info(
            "NGRC optimizer search start | run=%s optimizers=%s n_calls=%d",
            cfg["experiment"]["run_id"],
            ",".join(name for name, _ in optimizer_entries),
            n_calls,
        )

        best_rows = []

        for optimizer_name, optimizer_seed in optimizer_entries:
            best_row = _run_one_optimizer(
                cfg=cfg,
                optimizer_name=optimizer_name,
                # skopt's GP path normalizes dimensions in place (Dimension.set_transformer),
                # so each optimizer needs its own copy to avoid one optimizer's construction
                # silently changing another's sampling behavior within the same process.
                dimensions=copy.deepcopy(dimensions),
                n_calls=n_calls,
                n_initial_points=n_initial_points,
                random_state=optimizer_seed,
                search_out_dir=search_out_dir,
                max_feature_dimension=max_feature_dimension,
            )
            best_rows.append(best_row)

        best_rows_path = search_out_dir / "optimizer_best_summary.csv"
        best_fieldnames = [
            "optimizer",
            "candidate_run_id",
            "feature_set",
            "context_steps",
            "delay_spacing",
            "ridge_alpha",
            "quadratic_mode",
            "include_bias",
            "feature_dimension",
            "valid_relative_l2_horizon_time",
            "valid_relative_l2_mean",
            "finite_relative_l2_fraction",
            "output_dir",
        ]

        with best_rows_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=best_fieldnames)
            writer.writeheader()

            for row in best_rows:
                writer.writerow({key: row.get(key, "") for key in best_fieldnames})

        overall_best = sorted(best_rows, key=lambda row: _score_from_summary(row["_summary"]))[0]

        base_run_group = str(require_key(output_cfg, "run_group", "output")).strip("/")
        search_run_id = str(require_key(experiment_cfg, "run_id", "experiment"))

        overall_best_cfg = copy.deepcopy(overall_best["_config"])
        overall_best_cfg["experiment"]["run_id"] = f"{search_run_id}_overall_best"
        overall_best_cfg["experiment"]["type"] = "single"
        overall_best_cfg["output"]["run_group"] = f"{base_run_group}/{search_run_id}/overall_best"
        overall_best_cfg.pop("search", None)

        _write_json(search_out_dir / "overall_best_config_validation.json", overall_best_cfg)

        logger.info(
            "Overall best selected | optimizer=%s run=%s valid_h=%.6g mean=%s",
            overall_best["optimizer"],
            overall_best["candidate_run_id"],
            overall_best["valid_relative_l2_horizon_time"],
            overall_best["valid_relative_l2_mean"],
        )

        overall_best_config_path = search_out_dir / "overall_best_config_validation.json"

        evaluate_best_test = require_section(cfg, "search").get(
            "evaluate_best_test",
            False,
        )
        if not isinstance(evaluate_best_test, bool):
            raise TypeError(
                "search.evaluate_best_test must be a boolean if provided."
            )

        if append_best_to_registry and not evaluate_best_test:
            raise ValueError(
                "search.append_best_to_registry=true requires "
                "search.evaluate_best_test=true because registry rows include test metrics."
            )

        logger.info(
            "Overall best validation run | evaluate_test=%s append_registry=%s",
            str(evaluate_best_test).lower(),
            str(append_best_to_registry).lower(),
        )

        overall_result = run_ngrc_experiment(
            overall_best_cfg,
            repo_root=REPO_ROOT,
            config_path=overall_best_config_path,
            append_registry=append_best_to_registry,
            evaluate_test=evaluate_best_test,
        )

        overall_summary = _load_summary(overall_result.summary_path)
        _write_json(search_out_dir / "overall_best_summary.json", overall_summary)

        logger.info(
            "NGRC optimizer search done | search_dir=%s overall_summary=%s",
            _display_path(search_out_dir),
            _display_path(overall_result.summary_path),
        )

        return 0

    except Exception:
        logger.exception("NGRC optimizer search failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
