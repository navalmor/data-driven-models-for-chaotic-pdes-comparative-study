#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import traceback
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
from forecasting.config import ConfigDict, load_config, require_key, require_section, validate_experiment
from forecasting.runners.rc_runner import run_rc_experiment


logger = get_logger("forecasting.scripts.optimizer_search_rc")

OPTIMIZER_NAMES: tuple[str, ...] = ("gp", "dummy", "forest", "gbrt")


FIELDNAMES = [
    "trial",
    "status",
    "optimizer",
    "run_id",
    "reservoir_size",
    "sparsity",
    "spectral_radius",
    "input_scaling",
    "leaking_rate",
    "ridge_alpha",
    "washout_steps",
    "include_bias",
    "include_input_skip",
    "feature_dimension",
    "valid_relative_l2_horizon_time",
    "valid_relative_l2_mean",
    "finite_relative_l2_fraction",
    "first_unsafe_valid_native_index",
    "safe_valid_prefix_length",
    "objective_value",
    "summary_path",
    "output_dir",
    "error",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run validation-only RC optimizer search."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a JSON RC optimizer-search config.",
    )
    add_logging_arguments(parser)
    return parser.parse_args()


def _get_logging_level_from_config(cfg: ConfigDict) -> str:
    logging_cfg = require_section(cfg, "logging")
    level = str(require_key(logging_cfg, "level", "logging")).upper()

    allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if level not in allowed:
        raise ValueError(f"logging.level must be one of {sorted(allowed)}, got {level!r}.")

    return level


def _write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _safe_token(value: Any) -> str:
    text = str(value).lower()
    text = text.replace("-", "m").replace("+", "p").replace(".", "p")
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def _candidate_run_id(search_run_id: str, optimizer_name: str, trial: int, candidate: dict[str, Any]) -> str:
    return (
        f"{search_run_id}_{optimizer_name}_trial{trial:04d}"
        f"_n{candidate['reservoir_size']}"
        f"_sr{_safe_token(candidate['spectral_radius'])}"
        f"_in{_safe_token(candidate['input_scaling'])}"
        f"_leak{_safe_token(candidate['leaking_rate'])}"
        f"_ridge{_safe_token(candidate['ridge_alpha'])}"
        f"_w{candidate['washout_steps']}"
    )


def _make_dimension(name: str, spec: dict[str, Any]):
    kind = str(require_key(spec, "type", f"search.space.{name}"))

    if kind == "categorical":
        values = require_key(spec, "values", f"search.space.{name}")
        if not isinstance(values, list) or not values:
            raise ValueError(f"search.space.{name}.values must be a non-empty list.")
        return Categorical(values, name=name)

    if kind == "integer":
        low = int(require_key(spec, "low", f"search.space.{name}"))
        high = int(require_key(spec, "high", f"search.space.{name}"))
        return Integer(low, high, name=name)

    if kind == "real":
        low = float(require_key(spec, "low", f"search.space.{name}"))
        high = float(require_key(spec, "high", f"search.space.{name}"))
        return Real(low, high, prior="uniform", name=name)

    if kind == "real_log":
        low = float(require_key(spec, "low", f"search.space.{name}"))
        high = float(require_key(spec, "high", f"search.space.{name}"))
        return Real(low, high, prior="log-uniform", name=name)

    raise ValueError(f"Unsupported dimension type for {name}: {kind!r}")


def _build_dimensions(cfg: ConfigDict):
    space_cfg = require_section(require_section(cfg, "search"), "space")
    names = [
        "reservoir_size",
        "sparsity",
        "spectral_radius",
        "input_scaling",
        "leaking_rate",
        "ridge_alpha",
        "washout_steps",
    ]
    dimensions = [_make_dimension(name, require_section(space_cfg, name)) for name in names]
    return names, dimensions


def _point_to_candidate(names: list[str], point: list[Any]) -> dict[str, Any]:
    candidate = dict(zip(names, point))
    candidate["reservoir_size"] = int(candidate["reservoir_size"])
    candidate["sparsity"] = float(candidate["sparsity"])
    candidate["spectral_radius"] = float(candidate["spectral_radius"])
    candidate["input_scaling"] = float(candidate["input_scaling"])
    candidate["leaking_rate"] = float(candidate["leaking_rate"])
    candidate["ridge_alpha"] = float(candidate["ridge_alpha"])
    candidate["washout_steps"] = int(candidate["washout_steps"])
    return candidate


def _feature_dimension(candidate: dict[str, Any], *, data_mode: str) -> int:
    input_dim = 64 if data_mode == "full64" else 8
    return 1 + input_dim + int(candidate["reservoir_size"])


def _search_output_base(optimizer_cfg: ConfigDict) -> str:
    """Root directory for per-trial run outputs.

    Honours ``output.base_dir`` from the supplied search config when present, so example
    and user configs can direct trials to a chosen location. Falls back to a clean,
    gitignored default that keeps trial artifacts out of the frozen package.
    """
    output = optimizer_cfg.get("output")
    if isinstance(output, dict):
        base = output.get("base_dir")
        if isinstance(base, str) and base.strip():
            return base
    return "outputs/optimizer_search"


def _build_single_config(
    *,
    optimizer_cfg: ConfigDict,
    search_run_id: str,
    data_mode: str,
    optimizer_name: str,
    candidate: dict[str, Any],
    trial: int,
) -> ConfigDict:
    run_id = _candidate_run_id(search_run_id, optimizer_name, trial, candidate)

    return {
        "experiment": {
            "run_id": run_id,
            "type": "single",
            "model": "rc",
            "data_mode": data_mode,
        },
        "data": dict(require_section(optimizer_cfg, "data")),
        "preprocessing": dict(require_section(optimizer_cfg, "preprocessing")),
        "model": {
            "reservoir_size": int(candidate["reservoir_size"]),
            "sparsity": float(candidate["sparsity"]),
            "spectral_radius": float(candidate["spectral_radius"]),
            "input_scaling": float(candidate["input_scaling"]),
            "leaking_rate": float(candidate["leaking_rate"]),
            "ridge_alpha": float(candidate["ridge_alpha"]),
            "washout_steps": int(candidate["washout_steps"]),
            "include_bias": True,
            "include_input_skip": True,
            "random_seed": int(candidate["random_seed"]),
        },
        "evaluation": dict(require_section(optimizer_cfg, "evaluation")),
        "output": {
            "base_dir": _search_output_base(optimizer_cfg),
            "overwrite": True,
            "run_group": f"search/optimizer/{search_run_id}/{optimizer_name}",
            "append_registry": False,
            "evaluate_test": False,
        },
        "logging": dict(require_section(optimizer_cfg, "logging")),
        "metadata": {
            "stage": "rc_v001_optimizer_search",
            "optimizer_search_run_id": search_run_id,
            "optimizer": optimizer_name,
            "trial": trial,
        },
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_existing_rows(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}

    rows: dict[int, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                trial = int(row["trial"])
            except Exception:
                continue
            rows[trial] = row

    return rows


def _objective_from_metrics(horizon: float, mean: float) -> float:
    if not math.isfinite(horizon):
        return 1.0e9
    if not math.isfinite(mean):
        mean = 1.0e6
    return -float(horizon) + 1.0e-6 * float(mean)


def _row_from_result(
    *,
    trial: int,
    optimizer_name: str,
    candidate: dict[str, Any],
    result: Any,
) -> dict[str, Any]:
    summary = _read_json(result.summary_path)
    valid = summary["validation_metrics_physical"]
    params = summary["rc_parameters"]

    horizon = float(valid["relative_l2_horizon_time"])
    mean = float(valid["relative_l2_mean"])
    objective = _objective_from_metrics(horizon, mean)

    return {
        "trial": trial,
        "status": "ok",
        "optimizer": optimizer_name,
        "run_id": summary["run_id"],
        "reservoir_size": params["reservoir_size"],
        "sparsity": params["sparsity"],
        "spectral_radius": params["spectral_radius"],
        "input_scaling": params["input_scaling"],
        "leaking_rate": params["leaking_rate"],
        "ridge_alpha": params["ridge_alpha"],
        "washout_steps": params["washout_steps"],
        "include_bias": params["include_bias"],
        "include_input_skip": params["include_input_skip"],
        "feature_dimension": summary["feature_dimension"],
        "valid_relative_l2_horizon_time": horizon,
        "valid_relative_l2_mean": mean,
        "finite_relative_l2_fraction": valid["finite_relative_l2_fraction"],
        "first_unsafe_valid_native_index": summary.get("first_unsafe_valid_native_index"),
        "safe_valid_prefix_length": summary.get("safe_valid_prefix_length"),
        "objective_value": objective,
        "summary_path": str(result.summary_path),
        "output_dir": str(result.output_dir),
        "error": "",
    }


def _failed_row(
    *,
    trial: int,
    optimizer_name: str,
    candidate: dict[str, Any],
    feature_dimension: int,
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "trial": trial,
        "status": "failed",
        "optimizer": optimizer_name,
        "run_id": "",
        "reservoir_size": candidate["reservoir_size"],
        "sparsity": candidate["sparsity"],
        "spectral_radius": candidate["spectral_radius"],
        "input_scaling": candidate["input_scaling"],
        "leaking_rate": candidate["leaking_rate"],
        "ridge_alpha": candidate["ridge_alpha"],
        "washout_steps": candidate["washout_steps"],
        "include_bias": True,
        "include_input_skip": True,
        "feature_dimension": feature_dimension,
        "valid_relative_l2_horizon_time": "",
        "valid_relative_l2_mean": "",
        "finite_relative_l2_fraction": "",
        "first_unsafe_valid_native_index": "",
        "safe_valid_prefix_length": "",
        "objective_value": 1.0e9,
        "summary_path": "",
        "output_dir": "",
        "error": repr(exc) + "\n" + traceback.format_exc(limit=3),
    }


def _is_better(row: dict[str, Any], best_row: dict[str, Any] | None) -> bool:
    if row.get("status") != "ok":
        return False

    if best_row is None:
        return True

    horizon = float(row["valid_relative_l2_horizon_time"])
    mean = float(row["valid_relative_l2_mean"])
    best_horizon = float(best_row["valid_relative_l2_horizon_time"])
    best_mean = float(best_row["valid_relative_l2_mean"])

    return (horizon > best_horizon) or (
        math.isclose(horizon, best_horizon) and mean < best_mean
    )


def _base_estimator_name(optimizer_name: str) -> str:
    mapping = {
        "gp": "GP",
        "dummy": "DUMMY",
        "forest": "RF",
        "gbrt": "GBRT",
    }
    if optimizer_name not in mapping:
        raise ValueError(f"Unsupported optimizer {optimizer_name!r}; expected one of {sorted(mapping)}.")
    return mapping[optimizer_name]


def _existing_points_and_values(
    *,
    names: list[str],
    rows_by_trial: dict[int, dict[str, Any]],
) -> tuple[list[list[Any]], list[float]]:
    x_values: list[list[Any]] = []
    y_values: list[float] = []

    for trial in sorted(rows_by_trial):
        row = rows_by_trial[trial]
        if row.get("status") != "ok":
            continue

        x_values.append([
            int(row["reservoir_size"]),
            float(row["sparsity"]),
            float(row["spectral_radius"]),
            float(row["input_scaling"]),
            float(row["leaking_rate"]),
            float(row["ridge_alpha"]),
            int(row["washout_steps"]),
        ])
        y_values.append(float(row["objective_value"]))

    return x_values, y_values


def _run_one_optimizer(
    *,
    cfg: ConfigDict,
    search_run_id: str,
    data_mode: str,
    optimizer_name: str,
    optimizer_index: int,
    optimizer_seed: int,
    names: list[str],
    dimensions: list[Any],
    search_out_dir: Path,
) -> dict[str, Any]:
    search_cfg = require_section(cfg, "search")

    n_calls = int(require_key(search_cfg, "n_calls", "search"))
    n_initial_points = int(search_cfg.get("n_initial_points", min(10, n_calls)))
    random_seed_base = int(require_key(search_cfg, "random_seed_base", "search"))
    max_feature_dimension = int(search_cfg.get("max_feature_dimension", 20000))
    resume = bool(search_cfg.get("resume", True))

    optimizer_dir = search_out_dir / optimizer_name
    optimizer_dir.mkdir(parents=True, exist_ok=True)

    results_path = optimizer_dir / "optimizer_results.csv"
    existing_rows = _read_existing_rows(results_path) if resume else {}
    rows_by_trial = dict(existing_rows)

    skopt_optimizer = Optimizer(
        dimensions=dimensions,
        base_estimator=_base_estimator_name(optimizer_name),
        n_initial_points=n_initial_points,
        acq_func="EI",
        random_state=optimizer_seed,
    )

    existing_x, existing_y = _existing_points_and_values(names=names, rows_by_trial=rows_by_trial)
    if existing_x:
        skopt_optimizer.tell(existing_x, existing_y)

    best_row: dict[str, Any] | None = None
    best_cfg: ConfigDict | None = None

    for row in rows_by_trial.values():
        if _is_better(row, best_row):
            best_row = row

    logger.info(
        "Optimizer start | run=%s optimizer=%s calls=%d existing=%d",
        search_run_id,
        optimizer_name,
        n_calls,
        len(existing_rows),
    )

    for trial in range(1, n_calls + 1):
        if trial in existing_rows and existing_rows[trial].get("status") in {"ok", "failed"}:
            logger.info(
                "Resume skip | optimizer=%s trial=%d status=%s run=%s",
                optimizer_name,
                trial,
                existing_rows[trial].get("status"),
                existing_rows[trial].get("run_id", ""),
            )
            continue

        point = skopt_optimizer.ask()
        candidate = _point_to_candidate(names, point)
        candidate["random_seed"] = random_seed_base + 100000 * optimizer_index + trial

        feature_dimension = _feature_dimension(candidate, data_mode=data_mode)

        logger.info(
            "Trial | optimizer=%s %d/%d n=%s sparsity=%s sr=%.6g input=%.6g leak=%.6g ridge=%.6g washout=%s",
            optimizer_name,
            trial,
            n_calls,
            candidate["reservoir_size"],
            candidate["sparsity"],
            candidate["spectral_radius"],
            candidate["input_scaling"],
            candidate["leaking_rate"],
            candidate["ridge_alpha"],
            candidate["washout_steps"],
        )

        if feature_dimension > max_feature_dimension:
            objective_value = 1.0e9
            row = _failed_row(
                trial=trial,
                optimizer_name=optimizer_name,
                candidate=candidate,
                feature_dimension=feature_dimension,
                exc=ValueError(
                    f"feature_dimension {feature_dimension} > max_feature_dimension {max_feature_dimension}"
                ),
            )
            rows_by_trial[trial] = row
            skopt_optimizer.tell(point, objective_value)
            _write_rows(results_path, [rows_by_trial[k] for k in sorted(rows_by_trial)])
            continue

        candidate_cfg = _build_single_config(
            optimizer_cfg=cfg,
            search_run_id=search_run_id,
            data_mode=data_mode,
            optimizer_name=optimizer_name,
            candidate=candidate,
            trial=trial,
        )

        try:
            result = run_rc_experiment(
                candidate_cfg,
                repo_root=REPO_ROOT,
                config_path=None,
                append_registry=False,
                evaluate_test=False,
            )
            row = _row_from_result(
                trial=trial,
                optimizer_name=optimizer_name,
                candidate=candidate,
                result=result,
            )
            objective_value = float(row["objective_value"])

        except Exception as exc:
            logger.exception("Trial failed | optimizer=%s trial=%d", optimizer_name, trial)
            row = _failed_row(
                trial=trial,
                optimizer_name=optimizer_name,
                candidate=candidate,
                feature_dimension=feature_dimension,
                exc=exc,
            )
            objective_value = 1.0e9

        rows_by_trial[trial] = row
        skopt_optimizer.tell(point, objective_value)
        _write_rows(results_path, [rows_by_trial[k] for k in sorted(rows_by_trial)])

        if _is_better(row, best_row):
            best_row = row
            best_cfg = candidate_cfg
            logger.info(
                "New optimizer best | optimizer=%s trial=%d valid_h=%s valid_mean=%s run=%s",
                optimizer_name,
                trial,
                row["valid_relative_l2_horizon_time"],
                row["valid_relative_l2_mean"],
                row["run_id"],
            )

    rows = [rows_by_trial[k] for k in sorted(rows_by_trial)]
    _write_rows(results_path, rows)

    if best_row is None:
        raise RuntimeError(f"No successful trials for optimizer {optimizer_name}.")

    if best_cfg is None:
        # This happens when all best rows were loaded from resume.
        best_trial = int(best_row["trial"])
        candidate = {
            "reservoir_size": int(best_row["reservoir_size"]),
            "sparsity": float(best_row["sparsity"]),
            "spectral_radius": float(best_row["spectral_radius"]),
            "input_scaling": float(best_row["input_scaling"]),
            "leaking_rate": float(best_row["leaking_rate"]),
            "ridge_alpha": float(best_row["ridge_alpha"]),
            "washout_steps": int(best_row["washout_steps"]),
            "random_seed": random_seed_base + 100000 * optimizer_index + best_trial,
        }
        best_cfg = _build_single_config(
            optimizer_cfg=cfg,
            search_run_id=search_run_id,
            data_mode=data_mode,
            optimizer_name=optimizer_name,
            candidate=candidate,
            trial=best_trial,
        )

    _write_json(optimizer_dir / "best_config_validation.json", best_cfg)
    _write_json(optimizer_dir / "best_summary.json", best_row)

    ok = sum(1 for row in rows if row.get("status") == "ok")
    failed = sum(1 for row in rows if row.get("status") == "failed")

    logger.info(
        "Optimizer done | optimizer=%s rows=%d ok=%d failed=%d best=%s valid_h=%s",
        optimizer_name,
        len(rows),
        ok,
        failed,
        best_row["run_id"],
        best_row["valid_relative_l2_horizon_time"],
    )

    return best_row


def main() -> int:
    args = _parse_args()

    try:
        cfg = load_config(args.config)
        configure_from_args(args, component="OPT-RC",
                            config_level=_get_logging_level_from_config(cfg))
        validate_experiment(cfg, expected_type="rc_optimizer_search", expected_model="rc")

        experiment_cfg = require_section(cfg, "experiment")
        search_cfg = require_section(cfg, "search")

        search_run_id = str(require_key(experiment_cfg, "run_id", "experiment"))
        data_mode = str(require_key(experiment_cfg, "data_mode", "experiment"))
        optimizer_seeds_cfg = require_section(search_cfg, "optimizer_seeds")

        names, dimensions = _build_dimensions(cfg)

        base_dir = Path(_search_output_base(cfg))
        if not base_dir.is_absolute():
            base_dir = REPO_ROOT / base_dir
        search_out_dir = (
            base_dir
            / "rc"
            / "search"
            / "optimizer"
            / search_run_id
        )
        search_out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "RC optimizer search start | run=%s mode=%s optimizers=%s",
            search_run_id,
            data_mode,
            ",".join(OPTIMIZER_NAMES),
        )

        per_optimizer_best: list[dict[str, Any]] = []
        overall_best: dict[str, Any] | None = None

        for optimizer_index, optimizer_name in enumerate(OPTIMIZER_NAMES, start=1):
            optimizer_seed = int(require_key(optimizer_seeds_cfg, optimizer_name, "search.optimizer_seeds"))
            best = _run_one_optimizer(
                cfg=cfg,
                search_run_id=search_run_id,
                data_mode=data_mode,
                optimizer_name=optimizer_name,
                optimizer_index=optimizer_index,
                optimizer_seed=optimizer_seed,
                names=names,
                dimensions=dimensions,
                search_out_dir=search_out_dir,
            )
            per_optimizer_best.append(best)

            if _is_better(best, overall_best):
                overall_best = best

        if overall_best is None:
            raise RuntimeError("No successful optimizer result found.")

        # Reconstruct selected config from the overall best row.
        best_trial = int(overall_best["trial"])
        best_optimizer = str(overall_best["optimizer"])
        candidate = {
            "reservoir_size": int(overall_best["reservoir_size"]),
            "sparsity": float(overall_best["sparsity"]),
            "spectral_radius": float(overall_best["spectral_radius"]),
            "input_scaling": float(overall_best["input_scaling"]),
            "leaking_rate": float(overall_best["leaking_rate"]),
            "ridge_alpha": float(overall_best["ridge_alpha"]),
            "washout_steps": int(overall_best["washout_steps"]),
            "random_seed": int(require_key(search_cfg, "random_seed_base", "search")) + 100000 * (OPTIMIZER_NAMES.index(best_optimizer) + 1) + best_trial,
        }

        overall_best_cfg = _build_single_config(
            optimizer_cfg=cfg,
            search_run_id=search_run_id,
            data_mode=data_mode,
            optimizer_name=best_optimizer,
            candidate=candidate,
            trial=best_trial,
        )

        _write_json(search_out_dir / "per_optimizer_best_summary.json", per_optimizer_best)
        _write_json(search_out_dir / "overall_best_summary.json", overall_best)
        _write_json(search_out_dir / "overall_best_config_validation.json", overall_best_cfg)

        logger.info(
            "RC optimizer search done | run=%s best_optimizer=%s best_run=%s valid_h=%s",
            search_run_id,
            overall_best["optimizer"],
            overall_best["run_id"],
            overall_best["valid_relative_l2_horizon_time"],
        )

        return 0

    except Exception:
        logger.exception("RC optimizer search failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
