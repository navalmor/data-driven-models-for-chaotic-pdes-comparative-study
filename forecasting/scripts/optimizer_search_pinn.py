#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import re
import sys
import traceback
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
FORECASTING_SRC = REPO_ROOT / "forecasting" / "src"
for _bootstrap_path in (REPO_ROOT, FORECASTING_SRC):
    if str(_bootstrap_path) not in sys.path:
        sys.path.insert(0, str(_bootstrap_path))

from skopt import Optimizer
from skopt.space import Categorical, Integer, Real

try:
    from skopt import dummy_minimize, forest_minimize, gbrt_minimize, gp_minimize
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"scikit-optimize is required for PINN optimizer search: {exc}") from exc

from common.logging_setup import add_logging_arguments, configure_from_args, get_logger
from forecasting.pinn_sweep_utils import read_json, slugify, write_json
from forecasting.runners.pinn_runner import run_pinn_experiment

logger = get_logger("forecasting.scripts.optimizer_search_pinn")


OPTIMIZER_NAMES: tuple[str, ...] = ("dummy", "gp", "gbrt", "forest")


# --------------------------------------------------------------------------
# full64: skopt functional API (dummy_minimize/gp_minimize/gbrt_minimize/
# forest_minimize) with a closure-based objective.
# Body preserved exactly as the former standalone optimizer_search_pinn.py.
# --------------------------------------------------------------------------

def _search_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    if "search" in cfg:
        return cfg["search"]
    if "optimizer_search" in cfg:
        return cfg["optimizer_search"]
    raise KeyError("Expected config section 'search'.")


def _resolve_optimizer_names(
    search_cfg: dict[str, Any],
    optimizer_settings: dict[str, Any],
) -> tuple[str, ...]:
    """
    Resolve which optimizer families this search run executes.

    Omitting search.optimizers runs all four families (dummy/gp/gbrt/forest),
    so existing configs are unaffected. Listing a subset runs only those, which
    lets the four families be scheduled as independent jobs; the union across
    those jobs must still cover all four before a winner is selected.
    """
    requested = search_cfg.get("optimizers", list(OPTIMIZER_NAMES))

    if not isinstance(requested, list) or not requested:
        raise ValueError(
            f"search.optimizers must be a non-empty list, got {requested!r}."
        )

    optimizer_names = tuple(str(name) for name in requested)

    unknown = sorted(set(optimizer_names) - set(OPTIMIZER_NAMES))
    if unknown:
        raise ValueError(
            f"Unknown optimizer(s) in search.optimizers: {unknown}. "
            f"Supported values are {sorted(OPTIMIZER_NAMES)}."
        )

    if len(set(optimizer_names)) != len(optimizer_names):
        raise ValueError(
            f"Duplicate optimizer(s) in search.optimizers: {list(optimizer_names)}."
        )

    missing_settings = [name for name in optimizer_names if name not in optimizer_settings]
    if missing_settings:
        raise ValueError(
            f"search.optimizer_settings is missing entries for: {missing_settings}."
        )

    return optimizer_names


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
    return value


def _slug_full64(value: Any) -> str:
    text = str(value)
    text = text.replace("-", "m").replace("+", "")
    text = text.replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _space_dimension(spec: dict[str, Any]):
    name = spec["name"]
    kind = spec["type"]
    if kind == "categorical":
        return Categorical(spec["values"], name=name)
    if kind == "integer":
        return Integer(int(spec["low"]), int(spec["high"]), name=name)
    if kind == "real":
        return Real(float(spec["low"]), float(spec["high"]), prior=spec.get("prior", "uniform"), name=name)
    raise ValueError(f"Unsupported search-space dimension type: {kind}")


def _extract_validation(summary: dict[str, Any]) -> dict[str, Any]:
    valid = (
        summary.get("validation_metrics_physical")
        or summary.get("valid_metrics_physical")
        or summary.get("validation_metrics")
        or {}
    )
    return {
        "valid_relative_l2_horizon_time": valid.get("relative_l2_horizon_time"),
        "valid_relative_l2_mean": valid.get("relative_l2_mean"),
        "valid_relative_l2_final": valid.get("relative_l2_final"),
        "valid_finite_relative_l2_fraction": valid.get("finite_relative_l2_fraction"),
        "first_unsafe_index": valid.get("first_unsafe_index"),
        "safe_prefix_length": valid.get("safe_prefix_length"),
        "feature_dimension": summary.get("feature_dimension"),
    }


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _sort_key_full64(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        -_as_float(row.get("valid_relative_l2_horizon_time")),
        _as_float(row.get("valid_relative_l2_mean"), 1.0e9),
        -_as_float(row.get("valid_finite_relative_l2_fraction")),
    )


def _objective_from_row(row: dict[str, Any]) -> float:
    h = _as_float(row.get("valid_relative_l2_horizon_time"))
    mean = _as_float(row.get("valid_relative_l2_mean"), 1.0e9)
    finite = _as_float(row.get("valid_finite_relative_l2_fraction"), 0.0)
    if not math.isfinite(h):
        return 1.0e9
    if not math.isfinite(mean):
        mean = 1.0e9
    return -h + 1.0e-4 * mean - 1.0e-3 * finite


def _result_fields() -> list[str]:
    return [
        "optimizer",
        "trial",
        "status",
        "objective",
        "run_id",
        "summary_path",
        "error",
        "valid_relative_l2_horizon_time",
        "valid_relative_l2_mean",
        "valid_relative_l2_final",
        "valid_finite_relative_l2_fraction",
        "first_unsafe_index",
        "safe_prefix_length",
        "feature_dimension",
        "hidden_dim",
        "depth",
        "activation",
        "learning_rate",
        "epochs",
        "batches_per_epoch",
        "batch_size",
        "multistep_horizon",
        "beta_multistep",
        "lambda_phy",
        "physics_residual",
        "weight_decay",
        "random_seed",
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_result_fields(), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _write_best(path: Path, rows: list[dict[str, Any]]) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    payload: dict[str, Any] = {
        "rows": len(rows),
        "ok": len(ok),
        "failed": len(rows) - len(ok),
        "best": None,
    }
    if ok:
        payload["best"] = sorted(ok, key=_sort_key_full64)[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_optimizer_best_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    best_by_opt: list[dict[str, Any]] = []
    for opt in ["dummy", "gp", "gbrt", "forest"]:
        opt_rows = [r for r in ok if r.get("optimizer") == opt]
        if opt_rows:
            best_by_opt.append(sorted(opt_rows, key=_sort_key_full64)[0])

    fields = [
        "optimizer",
        "trial",
        "run_id",
        "valid_relative_l2_horizon_time",
        "valid_relative_l2_mean",
        "valid_finite_relative_l2_fraction",
        "hidden_dim",
        "depth",
        "learning_rate",
        "epochs",
        "multistep_horizon",
        "beta_multistep",
        "lambda_phy",
        "summary_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in best_by_opt:
            writer.writerow(_jsonable(row))


def _write_best_config_validation(root: Path, rows: list[dict[str, Any]]) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    for opt in ["dummy", "gp", "gbrt", "forest"]:
        opt_rows = [r for r in ok if r.get("optimizer") == opt]
        if not opt_rows:
            continue
        best = sorted(opt_rows, key=_sort_key_full64)[0]
        summary_path = Path(str(best.get("summary_path", "")))
        trial_dir = summary_path.parent
        config_path = trial_dir / "config_resolved.json"
        if not config_path.exists():
            config_path = trial_dir / "config_input.json"

        payload: dict[str, Any] = {
            "optimizer": opt,
            "best_trial": _jsonable(best),
            "config_source": str(config_path),
            "config": None,
        }
        if config_path.exists():
            try:
                payload["config"] = json.loads(config_path.read_text())
            except Exception:
                payload["config"] = None

        out = root / opt / "best_config_validation.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _trial_run_id(search_run_id: str, optimizer: str, trial_index: int, model: dict[str, Any]) -> str:
    return (
        f"{search_run_id}_{optimizer}_trial{trial_index:04d}"
        f"_h{_slug_full64(model['hidden_dim'])}"
        f"_d{_slug_full64(model['depth'])}"
        f"_lr{_slug_full64(model['learning_rate'])}"
        f"_mh{_slug_full64(model['multistep_horizon'])}"
        f"_b{_slug_full64(model['beta_multistep'])}"
        f"_lam{_slug_full64(model['lambda_phy'])}"
    )


def _write_search_root_outputs(out_root: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(out_root / "optimizer_results.csv", rows)
    _write_best(out_root / "best_summary.json", rows)
    _write_optimizer_best_summary(out_root / "optimizer_best_summary.csv", rows)
    _write_best_config_validation(out_root, rows)


def run_optimizer(cfg: dict[str, Any], *, repo_root: Path, config_path: Path) -> None:
    experiment = cfg["experiment"]
    output_cfg = cfg["output"]
    opt_cfg = _search_cfg(cfg)

    assert output_cfg.get("evaluate_test") is False, "PINN optimizer search must be validation-only."
    assert output_cfg.get("append_registry") is False, "PINN optimizer search must not append registry."

    search_run_id = experiment["run_id"]
    base_model = copy.deepcopy(cfg["model"])
    base_dir = Path(output_cfg.get("base_dir", "experiments/forecasting"))
    out_root = repo_root / base_dir / "pinn" / "search" / "optimizer" / search_run_id
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "search_config_input.json").write_text(
        json.dumps(_jsonable(cfg), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    dims = [_space_dimension(spec) for spec in opt_cfg["space"]]
    dim_names = [dim.name for dim in dims]
    all_rows: list[dict[str, Any]] = []

    def evaluate(optimizer_name: str, trial_index: int, values: list[Any], optimizer_seed: int) -> float:
        params = {name: _jsonable(value) for name, value in zip(dim_names, values)}
        model = copy.deepcopy(base_model)
        model.update(params)
        model["activation"] = "tanh"
        model["residual_prediction"] = True
        model["physics_residual"] = "cnab2"
        model["physics_scale"] = "delta_rms"
        model["dealias"] = True
        model["random_seed"] = int(optimizer_seed + trial_index)

        run_id = _trial_run_id(search_run_id, optimizer_name, trial_index, model)
        group = f"search/optimizer/{search_run_id}/{optimizer_name}"

        trial_cfg = copy.deepcopy(cfg)
        trial_cfg["experiment"] = {**trial_cfg["experiment"], "run_id": run_id}
        trial_cfg["model"] = model
        trial_cfg["output"] = {
            **trial_cfg["output"],
            "evaluate_test": False,
            "append_registry": False,
            "overwrite": True,
        }
        trial_cfg["_runtime"] = {
            "run_id_override": run_id,
            "run_group_override": group,
        }

        trial_out = repo_root / base_dir / "pinn" / group / run_id
        summary_path = trial_out / "summary.json"

        row: dict[str, Any] = {
            "optimizer": optimizer_name,
            "trial": trial_index,
            "status": "failed",
            "objective": 1.0e9,
            "run_id": run_id,
            "summary_path": str(summary_path),
            "error": "",
            **model,
        }

        try:
            run_pinn_experiment(
                trial_cfg,
                repo_root=repo_root,
                config_path=config_path,
                append_registry=False,
                evaluate_test=False,
            )
            if not summary_path.exists():
                raise FileNotFoundError(f"summary not found: {summary_path}")
            summary = json.loads(summary_path.read_text())
            row.update(_extract_validation(summary))
            row["status"] = "ok"
            row["objective"] = _objective_from_row(row)
            logger.info(
                "optimizer_result | %s trial=%s | METRIC h(physical time)=%s mean=%s "
                "finite=%s objective=%s",
                optimizer_name, trial_index,
                row.get("valid_relative_l2_horizon_time"),
                row.get("valid_relative_l2_mean"),
                row.get("valid_finite_relative_l2_fraction"),
                row["objective"],
            )
        except Exception as exc:
            row["error"] = repr(exc) + "\n" + traceback.format_exc(limit=8)
            row["objective"] = 1.0e9
            logger.warning("optimizer_failed | %s trial=%s run=%s | %s",
                           optimizer_name, trial_index, run_id, repr(exc))

        all_rows.append(row)
        _write_search_root_outputs(out_root, all_rows)
        _write_csv(out_root / optimizer_name / "optimizer_results.csv", [r for r in all_rows if r["optimizer"] == optimizer_name])
        _write_best(out_root / optimizer_name / "best_summary.json", [r for r in all_rows if r["optimizer"] == optimizer_name])
        _write_best_config_validation(out_root, all_rows)

        return float(row["objective"])

    optimizer_functions = {
        "dummy": dummy_minimize,
        "gp": gp_minimize,
        "gbrt": gbrt_minimize,
        "forest": forest_minimize,
    }

    optimizer_settings = opt_cfg["optimizer_settings"]
    optimizer_names = _resolve_optimizer_names(opt_cfg, optimizer_settings)

    logger.info("PINN optimizer families | run=%s optimizers=%s",
                search_run_id, ",".join(optimizer_names))

    for optimizer_name in optimizer_names:
        settings = optimizer_settings[optimizer_name]

        n_calls = int(settings["n_calls"])
        n_initial_points = int(settings.get("n_initial_points", min(12, n_calls)))
        random_seed = int(settings["random_seed"])

        logger.info("PINN optimizer start | run=%s optimizer=%s calls=%d initial=%d seed=%d",
                    search_run_id, optimizer_name, n_calls, n_initial_points, random_seed)

        local_trial = {"count": 0}

        def objective(values: list[Any]) -> float:
            local_trial["count"] += 1
            return evaluate(optimizer_name, local_trial["count"], values, random_seed)

        if optimizer_name == "dummy":
            optimizer_functions[optimizer_name](
                objective,
                dims,
                n_calls=n_calls,
                random_state=random_seed,
            )
        else:
            optimizer_functions[optimizer_name](
                objective,
                dims,
                n_calls=n_calls,
                n_initial_points=min(n_initial_points, n_calls),
                random_state=random_seed,
            )

        optimizer_rows = [r for r in all_rows if r["optimizer"] == optimizer_name]
        _write_best(out_root / optimizer_name / "best_summary.json", optimizer_rows)
        logger.info("PINN optimizer done | optimizer=%s rows=%d ok=%d",
                    optimizer_name, len(optimizer_rows),
                    sum(1 for r in optimizer_rows if r["status"] == "ok"))

    _write_search_root_outputs(out_root, all_rows)

    best = json.loads((out_root / "best_summary.json").read_text()).get("best")
    logger.info(
        "PINN optimizer search done | rows=%d ok=%d failed=%d best=%s valid_h=%s",
        len(all_rows),
        sum(1 for r in all_rows if r["status"] == "ok"),
        sum(1 for r in all_rows if r["status"] != "ok"),
        None if best is None else best.get("run_id"),
        None if best is None else best.get("valid_relative_l2_horizon_time"),
    )


def _run_full64_search(args: argparse.Namespace) -> None:
    repo_root = Path.cwd()
    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text())

    assert cfg["experiment"]["model"] == "pinn"
    assert cfg["experiment"]["data_mode"] == "full64"
    assert cfg["output"]["evaluate_test"] is False
    assert cfg["output"]["append_registry"] is False

    opt_cfg = _search_cfg(cfg)
    dims = opt_cfg["space"]
    total = sum(int(v["n_calls"]) for v in opt_cfg["optimizer_settings"].values())

    logger.info("PINN optimizer config confirmed | run_id=%s total_calls=%d",
                cfg["experiment"]["run_id"], total)
    logger.info("optimizers=%s", list(OPTIMIZER_NAMES))
    logger.info("space=%s", [d["name"] for d in dims])

    if args.dry_run:
        return

    run_optimizer(cfg, repo_root=repo_root, config_path=config_path)


# --------------------------------------------------------------------------
# latent8: skopt Optimizer class with a manual ask/tell loop (decodes to
# physical space). Body preserved exactly as the former standalone
# optimizer_search_pinn_latent8.py.
# --------------------------------------------------------------------------

RESULT_FIELDS = [
    "rank",
    "status",
    "run_id",
    "name",
    "optimizer",
    "trial",
    "objective",
    "valid_h",
    "valid_mean",
    "valid_final",
    "valid_finite_fraction",
    "first_unsafe_index",
    "safe_prefix_length",
    "feature_dimension",
    "hidden_dim",
    "depth",
    "activation",
    "learning_rate",
    "epochs",
    "batches_per_epoch",
    "batch_size",
    "multistep_horizon",
    "beta_multistep",
    "lambda_phy",
    "physics_residual",
    "physics_scale",
    "random_seed",
    "error",
]


def _format_float_slug(value: float) -> str:
    text = f"{value:.3g}"
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def _dimensions(space_cfg: dict[str, Any]):
    return [
        Categorical([int(v) for v in space_cfg["hidden_dim"]], name="hidden_dim"),
        Integer(int(space_cfg["depth"][0]), int(space_cfg["depth"][1]), name="depth"),
        Real(float(space_cfg["learning_rate"][0]), float(space_cfg["learning_rate"][1]), prior="log-uniform", name="learning_rate"),
        Integer(int(space_cfg["multistep_horizon"][0]), int(space_cfg["multistep_horizon"][1]), name="multistep_horizon"),
        Real(float(space_cfg["beta_multistep"][0]), float(space_cfg["beta_multistep"][1]), name="beta_multistep"),
        Real(float(space_cfg["lambda_phy"][0]), float(space_cfg["lambda_phy"][1]), prior="log-uniform", name="lambda_phy"),
        Categorical([int(v) for v in space_cfg["epochs"]], name="epochs"),
    ]


def _x_to_model(dimensions, x: list[Any], base_model: dict[str, Any]) -> dict[str, Any]:
    values = {dim.name: val for dim, val in zip(dimensions, x)}
    model = copy.deepcopy(base_model)
    model.update(
        {
            "hidden_dim": int(values["hidden_dim"]),
            "depth": int(values["depth"]),
            "learning_rate": float(values["learning_rate"]),
            "multistep_horizon": int(values["multistep_horizon"]),
            "beta_multistep": float(values["beta_multistep"]),
            "lambda_phy": float(values["lambda_phy"]),
            "epochs": int(values["epochs"]),
        }
    )
    return model


def _model_to_name(model: dict[str, Any]) -> str:
    return (
        f"h{int(model['hidden_dim'])}"
        f"_d{int(model['depth'])}"
        f"_lr{_format_float_slug(float(model['learning_rate']))}"
        f"_mh{int(model['multistep_horizon'])}"
        f"_b{_format_float_slug(float(model['beta_multistep']))}"
        f"_lam{_format_float_slug(float(model['lambda_phy']))}"
        f"_ep{int(model['epochs'])}"
    )


def _candidate_cfg(
    base_cfg: dict[str, Any],
    search_id: str,
    global_index: int,
    optimizer_name: str,
    trial_index: int,
    model: dict[str, Any],
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.pop("optimizer_search", None)

    name = _model_to_name(model)
    run_id = f"{search_id}_{optimizer_name}_trial{trial_index:04d}_{slugify(name)}"

    cfg.setdefault("experiment", {})
    cfg["experiment"]["run_id"] = run_id
    cfg["experiment"]["type"] = "optimizer_search"
    cfg["experiment"]["stage"] = "optimizer"
    cfg["experiment"]["run_group"] = search_id
    cfg["experiment"]["data_mode"] = "latent8"
    cfg["experiment"]["model"] = "pinn"

    cfg.setdefault("output", {})
    cfg["output"]["run_group"] = f"search/optimizer/{search_id}"
    cfg["output"]["overwrite"] = True
    cfg["output"]["evaluate_test"] = False
    cfg["output"]["append_registry"] = False

    cfg["model"] = model

    cfg.setdefault("metadata", {})
    cfg["metadata"]["optimizer_search_id"] = search_id
    cfg["metadata"]["optimizer"] = optimizer_name
    cfg["metadata"]["trial"] = trial_index
    cfg["metadata"]["global_trial_index"] = global_index
    cfg["metadata"]["candidate_name"] = name
    cfg["metadata"]["test_policy"] = "evaluate_test=false and append_registry=false for optimizer search"
    cfg["metadata"]["selection_policy"] = "validation physical relative-L2 horizon only"

    return cfg


def _objective_from_metrics(metrics: dict[str, Any]) -> float:
    h = float(metrics.get("relative_l2_horizon_time", 0.0))
    mean = float(metrics.get("relative_l2_mean", 1e6))
    finite = float(metrics.get("finite_relative_l2_fraction", 0.0))
    if not math.isfinite(h):
        h = 0.0
    if not math.isfinite(mean):
        mean = 1e6
    if not math.isfinite(finite):
        finite = 0.0
    # Minimize. Horizon dominates; mean and finite fraction are tie-breakers.
    return -h + 1e-4 * min(mean, 1e4) - 1e-3 * finite


def _result_row(
    *,
    status: str,
    run_id: str,
    name: str,
    optimizer_name: str,
    trial_index: int,
    objective: float,
    cfg: dict[str, Any],
    summary: dict[str, Any] | None,
    error: str = "",
) -> dict[str, Any]:
    model = cfg.get("model", {})
    metrics = (summary or {}).get("validation_metrics_physical", {}) if summary else {}

    return {
        "rank": "",
        "status": status,
        "run_id": run_id,
        "name": name,
        "optimizer": optimizer_name,
        "trial": trial_index,
        "objective": objective,
        "valid_h": metrics.get("relative_l2_horizon_time", ""),
        "valid_mean": metrics.get("relative_l2_mean", ""),
        "valid_final": metrics.get("relative_l2_final", ""),
        "valid_finite_fraction": metrics.get("finite_relative_l2_fraction", ""),
        "first_unsafe_index": metrics.get("first_unsafe_index", ""),
        "safe_prefix_length": metrics.get("safe_prefix_length", ""),
        "feature_dimension": (summary or {}).get("feature_dimension", "") if summary else "",
        "hidden_dim": model.get("hidden_dim", ""),
        "depth": model.get("depth", ""),
        "activation": model.get("activation", ""),
        "learning_rate": model.get("learning_rate", ""),
        "epochs": model.get("epochs", ""),
        "batches_per_epoch": model.get("batches_per_epoch", ""),
        "batch_size": model.get("batch_size", ""),
        "multistep_horizon": model.get("multistep_horizon", ""),
        "beta_multistep": model.get("beta_multistep", ""),
        "lambda_phy": model.get("lambda_phy", ""),
        "physics_residual": model.get("physics_residual", ""),
        "physics_scale": model.get("physics_scale", ""),
        "random_seed": model.get("random_seed", ""),
        "error": error,
    }


def _sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    try:
        h = float(row["valid_h"])
    except Exception:
        h = -1.0
    try:
        mean = float(row["valid_mean"])
    except Exception:
        mean = float("inf")
    try:
        finite = float(row["valid_finite_fraction"])
    except Exception:
        finite = -1.0
    return (-h, mean, -finite)


def _write_partial(search_dir: Path, rows: list[dict[str, Any]]) -> None:
    partial = search_dir / "results_partial.csv"
    with partial.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    ranked = sorted(ok_rows, key=_sort_key)
    best = ranked[0] if ranked else None
    write_json(
        search_dir / "summary_partial.json",
        {
            "rows_done": len(rows),
            "num_ok": len(ok_rows),
            "num_failed": len([r for r in rows if r["status"] != "ok"]),
            "best_so_far": best,
        },
    )


def _sample_dummy(dimensions, rng: random.Random) -> list[Any]:
    x = []
    for dim in dimensions:
        if isinstance(dim, Categorical):
            x.append(rng.choice(list(dim.categories)))
        elif isinstance(dim, Integer):
            x.append(rng.randint(int(dim.low), int(dim.high)))
        elif isinstance(dim, Real):
            low, high = float(dim.low), float(dim.high)
            if getattr(dim, "prior", None) == "log-uniform":
                x.append(10 ** rng.uniform(math.log10(low), math.log10(high)))
            else:
                x.append(rng.uniform(low, high))
        else:
            raise TypeError(f"Unsupported dimension: {dim}")
    return x


def _run_latent8_search(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    config_path = (repo_root / args.config).resolve()
    base_cfg = read_json(config_path)

    search_cfg = base_cfg["optimizer_search"]
    search_id = str(search_cfg["search_id"])
    optimizer_settings = search_cfg["optimizer_settings"]
    dimensions = _dimensions(search_cfg["search_space"])

    base_dir = repo_root / base_cfg.get("output", {}).get("base_dir", "experiments/forecasting")
    search_dir = base_dir / "pinn" / "search" / "optimizer" / search_id
    search_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    global_index = 0

    logger.info(
        "PINN latent8 optimizer search id=%s optimizers=%s total_trials=%d",
        search_id, ",".join(OPTIMIZER_NAMES),
        sum(int(v["n_calls"]) for v in optimizer_settings.values()),
    )

    for optimizer_name in OPTIMIZER_NAMES:
        opt_spec = optimizer_settings[optimizer_name]
        n_calls = int(opt_spec["n_calls"])
        n_initial_points = int(opt_spec.get("n_initial_points", 0))
        seed = int(opt_spec["random_seed"])

        logger.info("optimizer %s n_calls=%d seed=%d", optimizer_name, n_calls, seed)

        skopt_optimizer = None
        rng = random.Random(seed)

        if optimizer_name != "dummy":
            base_estimator = {
                "gp": "GP",
                "gbrt": "GBRT",
                "forest": "RF",
            }[optimizer_name]
            skopt_optimizer = Optimizer(
                dimensions=dimensions,
                base_estimator=base_estimator,
                n_initial_points=n_initial_points,
                random_state=seed,
                acq_func="EI",
            )

        for trial_index in range(1, n_calls + 1):
            global_index += 1

            if optimizer_name == "dummy":
                x = _sample_dummy(dimensions, rng)
            else:
                assert skopt_optimizer is not None
                x = skopt_optimizer.ask()

            model = _x_to_model(dimensions, x, base_cfg["model"])
            cfg = _candidate_cfg(base_cfg, search_id, global_index, optimizer_name, trial_index, model)
            run_id = cfg["experiment"]["run_id"]
            name = cfg["metadata"]["candidate_name"]

            candidate_cfg_path = search_dir / f"{run_id}_config.json"
            write_json(candidate_cfg_path, cfg)

            logger.info(
                "trial %04d optimizer=%s trial=%04d/%d name=%s run_id=%s",
                global_index, optimizer_name, trial_index, n_calls, name, run_id,
            )

            try:
                result = run_pinn_experiment(
                    cfg,
                    repo_root=repo_root,
                    config_path=config_path.relative_to(repo_root),
                    append_registry=False,
                    evaluate_test=False,
                )
                summary = read_json(result.summary_path)
                assert summary["evaluate_test"] is False
                assert summary["append_registry"] is False
                assert summary["data_mode"] == "latent8"
                assert summary["decoded_to_physical"] is True
                assert summary["pinn"]["kse_physics"]["residual_space"] == "decoded_physical_full64"

                metrics = summary["validation_metrics_physical"]
                objective = _objective_from_metrics(metrics)

                row = _result_row(
                    status="ok",
                    run_id=run_id,
                    name=name,
                    optimizer_name=optimizer_name,
                    trial_index=trial_index,
                    objective=objective,
                    cfg=cfg,
                    summary=summary,
                )

                if skopt_optimizer is not None:
                    skopt_optimizer.tell(x, objective)

                logger.info(
                    "RESULT | METRIC objective=%s h(physical time)=%s mean=%s finite=%s first_unsafe=%s",
                    objective, row["valid_h"], row["valid_mean"],
                    row["valid_finite_fraction"], row["first_unsafe_index"],
                )

            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                (search_dir / f"{run_id}_error.txt").write_text(traceback.format_exc())
                objective = 1e6

                if skopt_optimizer is not None:
                    skopt_optimizer.tell(x, objective)

                row = _result_row(
                    status="failed",
                    run_id=run_id,
                    name=name,
                    optimizer_name=optimizer_name,
                    trial_index=trial_index,
                    objective=objective,
                    cfg=cfg,
                    summary=None,
                    error=err,
                )
                logger.warning("FAILED %s", err)

            rows.append(row)
            _write_partial(search_dir, rows)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    ranked = sorted(ok_rows, key=_sort_key)
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i

    failed = [r for r in rows if r["status"] != "ok"]
    output_rows = ranked + failed

    results_csv = search_dir / "results.csv"
    with results_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    best = ranked[0] if ranked else None

    by_optimizer = {}
    for name in OPTIMIZER_NAMES:
        opt_rows = [r for r in ranked if r["optimizer"] == name]
        by_optimizer[name] = opt_rows[0] if opt_rows else None

    summary = {
        "search_id": search_id,
        "config_path": str(config_path.relative_to(repo_root)),
        "output_dir": str(search_dir.relative_to(repo_root)),
        "optimizer_families": list(OPTIMIZER_NAMES),
        "trials_per_optimizer": {name: int(optimizer_settings[name]["n_calls"]) for name in OPTIMIZER_NAMES},
        "total_optimizer_trials": sum(int(v["n_calls"]) for v in optimizer_settings.values()),
        "num_ok": len(ok_rows),
        "num_failed": len(failed),
        "evaluate_test": False,
        "append_registry": False,
        "selection_metric": "validation physical relative_l2_horizon_time, descending; mean relative_l2 as tie-breaker",
        "best": best,
        "best_by_optimizer": by_optimizer,
        "results_csv": str(results_csv.relative_to(repo_root)),
    }
    write_json(search_dir / "summary.json", summary)

    logger.info("optimizer search complete | ok=%d failed=%d", len(ok_rows), len(failed))
    logger.info("results_csv %s", results_csv)
    logger.info("summary_json %s", search_dir / "summary.json")
    if best:
        logger.info(
            "BEST | rank=%s optimizer=%s trial=%s run_id=%s name=%s | "
            "METRIC h(physical time)=%s mean=%s finite=%s",
            best["rank"], best["optimizer"], best["trial"], best["run_id"], best["name"],
            best["valid_h"], best["valid_mean"], best["valid_finite_fraction"],
        )


# --------------------------------------------------------------------------
# Dispatch: data_mode is read from the config, exactly like every other
# (non-PINN) model's single script already does internally.
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a validation-only PINN optimizer search.")
    parser.add_argument("--config", required=True, help="Path to optimizer-search JSON config.")
    parser.add_argument("--dry-run", action="store_true", help="full64 only: validate config and log the search summary without running.")
    parser.add_argument("--repo-root", default=".", help="latent8 only: repo root used to resolve config/output paths.")
    add_logging_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_from_args(args, component="OPT-PINN")
    data_mode = json.loads(Path(args.config).read_text())["experiment"]["data_mode"]

    if data_mode == "full64":
        _run_full64_search(args)
    elif data_mode == "latent8":
        _run_latent8_search(args)
    else:
        raise ValueError(f"Unsupported PINN optimizer search data_mode: {data_mode!r}")


if __name__ == "__main__":
    main()
