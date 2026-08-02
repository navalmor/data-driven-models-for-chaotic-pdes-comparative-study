from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from forecasting.config import ConfigDict, require_key, require_section, validate_experiment
from forecasting.visualization.forecast_plots import (
    plot_relative_l2_horizon,
    plot_spatiotemporal_comparison,
)
from forecasting.visualization.io import (
    load_optimizer_results,
    prepare_output_dir,
    resolve_path,
    write_json,
)
from forecasting.visualization.search_plots import (
    build_parameter_specs,
    ok_rows,
    plot_optimizer_convergence,
    plot_sampling_matrix,
    plot_surrogate_landscape,
)
from forecasting.visualization.style import configure_matplotlib


logger = logging.getLogger(__name__)


def _require_bool(section: ConfigDict, key: str, section_name: str, default: bool | None = None) -> bool:
    value = section.get(key, default)

    if not isinstance(value, bool):
        raise TypeError(f"{section_name}.{key} must be boolean, got {value!r}.")

    return value


def _require_list(section: ConfigDict, key: str, section_name: str) -> list[Any]:
    value = require_key(section, key, section_name)

    if not isinstance(value, list) or not value:
        raise TypeError(f"{section_name}.{key} must be a non-empty list.")

    return value


def _safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in label).strip("_")


def _display_path(path: str | Path, repo_root: Path) -> str:
    path = Path(path)

    try:
        return str(path.resolve().relative_to(repo_root))
    except ValueError:
        return str(path)


def _forecast_time_axis_mode(
    cfg: ConfigDict, forecast_cfg: ConfigDict, cli_mode: str | None
) -> str:
    """Resolve the forecast display-time mode: CLI > plotting.time_axis_mode >
    legacy forecast_plots.use_lyapunov > default lyapunov.

    The legacy ``forecast_plots.use_lyapunov`` boolean is honoured only when
    neither the CLI flag nor ``plotting.time_axis_mode`` is given, so existing
    configs keep their behaviour without adding a field.
    """
    from common.plotting import resolve_time_axis_mode  # lazy: no module-level common import

    plotting = cfg.get("plotting") if isinstance(cfg, dict) else None
    config_value = plotting.get("time_axis_mode") if isinstance(plotting, dict) else None

    if cli_mode is None and config_value is None and "use_lyapunov" in forecast_cfg:
        return "lyapunov" if bool(forecast_cfg.get("use_lyapunov")) else "physical"

    return resolve_time_axis_mode(cli_value=cli_mode, config_value=config_value)


def _representation_from_label(label: str) -> str:
    """Derive the representation slug ('full64'/'latent8') from a run label."""
    tail = str(label).strip().lower().rsplit("_", 1)[-1]
    if tail in {"full64", "latent8"}:
        return tail
    if "latent" in tail:
        return "latent8"
    return "full64"


def run_visualization_config(
    cfg: ConfigDict,
    *,
    repo_root: str | Path,
    config_path: str | Path | None = None,
    time_axis_mode: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """
    Generate configured forecasting visualizations.

    Search diagnostics use optimizer-search validation metrics only.
    Forecast diagnostics use selected final-run outputs.

    ``time_axis_mode`` (``lyapunov``/``physical``/``None``) is the resolved CLI
    override for the forecast time axis; it is display-only and never applies to
    the optimiser-search horizon axes.

    ``output_root`` overrides ``output.base_dir`` so a caller can render previews
    outside the repository without editing a config. It changes only where the
    figures are written.
    """
    repo_root = Path(repo_root).expanduser().resolve()

    validate_experiment(cfg, expected_type="forecasting_visualization", expected_model="forecasting")
    configure_matplotlib()

    visualization_cfg = require_section(cfg, "visualization")
    inputs_cfg = require_section(cfg, "inputs")
    search_cfg = require_section(cfg, "search_plots")
    forecast_cfg = require_section(cfg, "forecast_plots")
    output_cfg = require_section(cfg, "output")

    run_id = str(require_key(visualization_cfg, "run_id", "visualization"))

    if output_root is not None:
        output_base = Path(output_root).expanduser().resolve()
    else:
        output_base = resolve_path(require_key(output_cfg, "base_dir", "output"), repo_root)
    overwrite = _require_bool(output_cfg, "overwrite", "output", default=False)
    formats = [str(fmt).lower().lstrip(".") for fmt in _require_list(output_cfg, "formats", "output")]
    dpi = int(output_cfg.get("dpi", 300))

    output_dir = prepare_output_dir(output_base, overwrite=overwrite)

    logger.info("Visualization start | run=%s output=%s", run_id, output_dir)

    index: dict[str, Any] = {
        "run_id": run_id,
        "config_path": (_display_path(Path(config_path), repo_root) if config_path is not None else None),
        "output_dir": _display_path(output_dir, repo_root),
        "figures": [],
        "skipped": [],
    }

    if _require_bool(search_cfg, "enabled", "search_plots", default=True):
        metric = str(require_key(search_cfg, "metric", "search_plots"))
        objective = str(search_cfg.get("objective", "maximize"))
        parameter_specs = build_parameter_specs(_require_list(search_cfg, "parameters", "search_plots"))

        # Config-driven Lyapunov-time axis, mirroring AESINDyConfig's
        # use_lyapunov/lambda_max fields. Lives here (plotting config) rather
        # than in any model/runner config, since it never touches a result file.
        use_lyapunov = bool(search_cfg.get("use_lyapunov", False))
        lambda_max = float(search_cfg.get("lambda_max", 1.0))

        surrogate_cfg = search_cfg.get("surrogate", {})
        if not isinstance(surrogate_cfg, dict):
            raise TypeError("search_plots.surrogate must be an object if provided.")

        for group in _require_list(inputs_cfg, "optimizer_groups", "inputs"):
            if not isinstance(group, dict):
                raise TypeError("Each inputs.optimizer_groups item must be an object.")

            label = str(require_key(group, "label", "optimizer_group"))
            search_dirs = [resolve_path(path, repo_root) for path in _require_list(group, "search_dirs", "optimizer_group")]

            rows = load_optimizer_results(search_dirs)
            rows = ok_rows(rows, metric)

            group_dir = output_dir / _safe_label(label)
            group_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Plot search group | label=%s rows=%d", label, len(rows))

            try:
                paths = plot_optimizer_convergence(
                    rows,
                    metric=metric,
                    objective=objective,
                    output_base=group_dir / "optimizer_convergence",
                    formats=formats,
                    dpi=dpi,
                    use_lyapunov=use_lyapunov,
                    lambda_max=lambda_max,
                )
                index["figures"].append({"label": label, "kind": "optimizer_convergence", "paths": [_display_path(path, repo_root) for path in paths]})
            except Exception as exc:
                logger.exception("Failed optimizer convergence plot | label=%s", label)
                index["skipped"].append({"label": label, "kind": "optimizer_convergence", "reason": repr(exc)})

            try:
                paths = plot_sampling_matrix(
                    rows,
                    specs=parameter_specs,
                    metric=metric,
                    objective=objective,
                    output_base=group_dir / "optimizer_sampling_matrix",
                    formats=formats,
                    dpi=dpi,
                    use_lyapunov=use_lyapunov,
                    lambda_max=lambda_max,
                )
                index["figures"].append({"label": label, "kind": "optimizer_sampling_matrix", "paths": [_display_path(path, repo_root) for path in paths]})
            except Exception as exc:
                logger.exception("Failed optimizer sampling matrix | label=%s", label)
                index["skipped"].append({"label": label, "kind": "optimizer_sampling_matrix", "reason": repr(exc)})

            if bool(surrogate_cfg.get("enabled", True)):
                try:
                    paths = plot_surrogate_landscape(
                        rows,
                        specs=parameter_specs,
                        metric=metric,
                        objective=objective,
                        output_base=group_dir / "optimizer_surrogate_landscape",
                        formats=formats,
                        dpi=dpi,
                        max_train_points=int(surrogate_cfg.get("max_train_points", 1200)),
                        random_seed=int(surrogate_cfg.get("random_seed", 0)),
                        x_parameter=surrogate_cfg.get("x_parameter"),
                        y_parameter=surrogate_cfg.get("y_parameter"),
                        use_lyapunov=use_lyapunov,
                        lambda_max=lambda_max,
                    )
                    index["figures"].append({"label": label, "kind": "optimizer_surrogate_landscape", "paths": [_display_path(path, repo_root) for path in paths]})
                except Exception as exc:
                    logger.exception("Failed optimizer surrogate landscape | label=%s", label)
                    index["skipped"].append({"label": label, "kind": "optimizer_surrogate_landscape", "reason": repr(exc)})

    if _require_bool(forecast_cfg, "enabled", "forecast_plots", default=True):
        splits = [str(split) for split in _require_list(forecast_cfg, "splits", "forecast_plots")]
        max_time_steps_raw = forecast_cfg.get("max_time_steps", 1500)
        max_time_steps = None if max_time_steps_raw is None else int(max_time_steps_raw)

        # Display-time axis: resolved via the shared time_axis_mode contract
        # (CLI > plotting.time_axis_mode > legacy forecast_plots.use_lyapunov >
        # lyapunov). Lives in the plotting config; it never touches a result file
        # and never changes which samples are shown.
        mode = _forecast_time_axis_mode(cfg, forecast_cfg, time_axis_mode)
        use_lyapunov = mode == "lyapunov"
        lambda_max = float(forecast_cfg.get("lambda_max", 1.0))
        logger.info("Forecast time axis | mode=%s", mode)

        for run in _require_list(inputs_cfg, "selected_runs", "inputs"):
            if not isinstance(run, dict):
                raise TypeError("Each inputs.selected_runs item must be an object.")

            label = str(require_key(run, "label", "selected_run"))
            model_name = str(require_key(run, "model_name", "selected_run"))
            representation = _representation_from_label(label)
            run_dir = resolve_path(require_key(run, "run_dir", "selected_run"), repo_root)

            run_out_dir = output_dir / _safe_label(label)
            run_out_dir.mkdir(parents=True, exist_ok=True)

            for split in splits:
                logger.info("Plot forecast run | label=%s split=%s", label, split)

                try:
                    paths = plot_relative_l2_horizon(
                        run_dir=run_dir,
                        split=split,
                        output_base=run_out_dir / f"{split}_relative_l2_horizon",
                        formats=formats,
                        dpi=dpi,
                        use_lyapunov=use_lyapunov,
                        lambda_max=lambda_max,
                    )
                    index["figures"].append({"label": label, "split": split, "kind": "relative_l2_horizon", "paths": [_display_path(path, repo_root) for path in paths]})
                except Exception as exc:
                    logger.exception("Failed relative-L2 horizon plot | label=%s split=%s", label, split)
                    index["skipped"].append({"label": label, "split": split, "kind": "relative_l2_horizon", "reason": repr(exc)})

                try:
                    paths = plot_spatiotemporal_comparison(
                        run_dir=run_dir,
                        split=split,
                        model_name=model_name,
                        representation=representation,
                        output_base=run_out_dir / f"{split}_spatiotemporal_comparison",
                        formats=formats,
                        dpi=dpi,
                        max_time_steps=max_time_steps,
                        use_lyapunov=use_lyapunov,
                        lambda_max=lambda_max,
                    )
                    index["figures"].append({"label": label, "split": split, "kind": "spatiotemporal_comparison", "paths": [_display_path(path, repo_root) for path in paths]})
                except Exception as exc:
                    logger.exception("Failed spatiotemporal plot | label=%s split=%s", label, split)
                    index["skipped"].append({"label": label, "split": split, "kind": "spatiotemporal_comparison", "reason": repr(exc)})

    write_json(output_dir / "figure_index.json", index)

    logger.info(
        "Visualization done | figures=%d skipped=%d index=%s",
        len(index["figures"]),
        len(index["skipped"]),
        output_dir / "figure_index.json",
    )

    return index
