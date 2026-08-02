from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from forecasting.visualization.io import (
    load_run_config,
    load_split_arrays,
    load_split_errors,
    load_split_metrics,
)
from forecasting.visualization.style import save_figure


def _cp():
    """Lazy accessor for shared plotting helpers (see runner._cp)."""
    from common import plotting

    return plotting


def _errors_to_arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray([float(row["time"]) for row in rows], dtype=float)

    values = []
    for row in rows:
        text = str(row.get("relative_l2", "")).strip()
        values.append(float(text) if text else float("nan"))

    rel_l2 = np.asarray(values, dtype=float)
    return times, rel_l2


def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)

    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def _checked_dt(value: Any, *, source: str) -> float:
    """Validate a candidate sampling interval, naming where it came from."""
    try:
        dt = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Sampling interval dt from {source} is not numeric: {value!r}") from None

    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Sampling interval dt from {source} must be finite and positive, got {dt!r}")

    return dt


def _resolve_dt(
    *,
    run_dir: str | Path,
    metrics: dict[str, Any],
    explicit_dt: float | None = None,
) -> float:
    """
    Resolve the physical sampling interval used to build a display time axis.

    Precedence, most to least authoritative:

      1. ``explicit_dt``           caller override (``None`` means "not supplied")
      2. ``metrics["dt"]``         written by ``forecasting.metrics.evaluate_forecast``
      3. ``config_resolved.json``  ``evaluation.dt``, the input that produced those metrics

    There is deliberately no numeric default. A silent fallback rescales the
    entire time axis while leaving every visible label intact, so a run that
    records no timing metadata anywhere is raised as an error rather than drawn
    on an assumed interval. Nothing here is model-aware: a leaner metrics schema
    simply falls through to the run config that produced it.
    """
    if explicit_dt is not None:
        return _checked_dt(explicit_dt, source="the explicit dt argument")

    if metrics.get("dt") is not None:
        return _checked_dt(metrics["dt"], source="metrics_physical.json")

    try:
        config: dict[str, Any] | None = load_run_config(run_dir)
    except FileNotFoundError:
        config = None

    if isinstance(config, dict):
        evaluation = config.get("evaluation")
        if isinstance(evaluation, dict) and evaluation.get("dt") is not None:
            return _checked_dt(evaluation["dt"], source="config_resolved.json evaluation.dt")

    raise ValueError(
        f"Cannot resolve the display sampling interval dt for run {run_dir}: no explicit dt "
        f"argument was given, metrics_physical.json has no 'dt', and "
        f"{'config_resolved.json has no evaluation.dt' if config is not None else 'config_resolved.json is missing'}."
    )


def _resolve_masked_from_index(metrics: dict[str, Any]) -> float | None:
    """
    Index from which the prediction is masked as unsafe, or ``None``.

    ``prediction_masked_from_index`` is the canonical field and stays
    authoritative wherever it carries a usable value. The PINN runner writes a
    leaner metrics schema that records the same quantity as
    ``first_unsafe_index``, so that name is read only as a fallback. This is a
    read-side alias: neither field is written, and ``metrics`` is not mutated.
    """
    for key in ("prediction_masked_from_index", "first_unsafe_index"):
        value = _metric_value(metrics, key)
        if value is not None and math.isfinite(value) and value >= 0.0:
            return value

    return None


# Lyapunov-time axis labels, mirroring AESINDyConfig's use_lyapunov/lambda_max
# convention. These plots read the toggle from the visualization config (not
# from any model/runner config), since lambda_max is a fixed physical constant
# shared by every run in a given figure set and this keeps relabeling a
# plotting-only concern with zero risk of perturbing locked result files.
LYAPUNOV_TIME_LABEL = r"time $t$ [$1/\lambda_{\max}$]"
PHYSICAL_TIME_LABEL = r"time $t$"

# Sizes, type sizes, colours, marker styles and notation all come from the
# central style contract (common/plot_style.json). Nothing in this module
# chooses a font size, a colour or a symbol on its own: a figure family names
# itself and the contract answers. See common/plot_style.py for why figure size
# is part of the style contract rather than a per-call-site detail.
def _ps():
    """Lazy accessor for the central style contract (see ``_cp``).

    ``common`` is imported inside the call, not at module import time, so this
    library package never assumes the repository root is on ``sys.path``.
    """
    from common import plot_style

    return plot_style


_FAMILY_ERROR_CURVE = "relative_l2_horizon"
_FAMILY_TRIPTYCH = "spatiotemporal_comparison"


def _time_axis(use_lyapunov: bool, lambda_max: float) -> tuple[float, str]:
    if use_lyapunov:
        return float(lambda_max), LYAPUNOV_TIME_LABEL
    return 1.0, PHYSICAL_TIME_LABEL


def _pred_symbol(model_name: str, representation: str) -> str:
    # Representation-aware prediction symbol delegated to the shared helper: the
    # full-state suffix and the latent suffix are chosen from the representation,
    # so a latent-space forecast is never given the full-state suffix.
    return _cp().prediction_symbol(model_name, representation)


def plot_relative_l2_horizon(
    *,
    run_dir: Path,
    split: str,
    output_base: Path,
    formats: list[str],
    dpi: int,
    use_lyapunov: bool = False,
    lambda_max: float = 1.0,
) -> list[str]:
    rows = load_split_errors(run_dir, split)
    metrics = load_split_metrics(run_dir, split)

    times, rel_l2 = _errors_to_arrays(rows)

    time_scale, x_label = _time_axis(use_lyapunov, lambda_max)
    times = times * time_scale

    threshold = _metric_value(metrics, "relative_l2_threshold")
    horizon = _metric_value(metrics, "relative_l2_horizon_time")
    horizon_scaled = None if horizon is None else horizon * time_scale
    masked_from = _metric_value(metrics, "prediction_masked_from_index")

    series = _ps().marker("primary_series")
    thr = _ps().marker("error_threshold")
    hor = _ps().marker("prediction_horizon")
    cut = _ps().marker("safety_cutoff")
    span = _ps().marker("masked_span")

    with plt.rc_context(_ps().rc_params(_FAMILY_ERROR_CURVE)):
        fig, ax = plt.subplots(figsize=_ps().figure_size_in(_FAMILY_ERROR_CURVE),
                               constrained_layout=True)

        ax.plot(times, rel_l2, color=series["color"], linewidth=series["linewidth"],
                label=r"Relative $L^2$ error")

        if threshold is not None:
            ax.axhline(threshold, linestyle=thr["linestyle"],
                       linewidth=thr["linewidth"], color=thr["color"],
                       label=rf"{_ps().term('error_threshold')}, "
                             rf"$\varepsilon = {threshold:g}$")

        if horizon_scaled is not None:
            # One appearance per meaning: a dashed line is the prediction
            # horizon here and on the triptych pages of the same figure.
            ax.axvline(horizon_scaled, linestyle=hor["linestyle"],
                       linewidth=hor["linewidth"], color=hor["color"],
                       label=_ps().horizon_label(horizon_scaled, use_lyapunov))

        if masked_from is not None and math.isfinite(masked_from):
            masked_from_int = int(masked_from)
            if 0 <= masked_from_int < len(times):
                cutoff_time = float(times[masked_from_int])
                ax.axvline(cutoff_time, linestyle=cut["linestyle"],
                           linewidth=cut["linewidth"], color=cut["color"],
                           alpha=cut["alpha"],
                           label=_ps().cutoff_label(cutoff_time, use_lyapunov))
                ax.axvspan(cutoff_time, times[-1], color=span["color"],
                           alpha=span["alpha"])

        ax.set_xlabel(x_label)
        ax.set_ylabel(_ps().axis_label(r"Relative $L^2$ error"))
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        # Explicit placement. loc="best" moves per case and can land on the data;
        # the error series rises to the right, so the upper-right corner is the
        # region it does not occupy early on.
        ax.legend(loc=_ps().family(_FAMILY_ERROR_CURVE)["legend_loc"], frameon=True)

    return save_figure(fig, output_base, formats, dpi)


def _downsample_time(arr: np.ndarray, max_time_steps: int | None) -> tuple[np.ndarray, int]:
    if max_time_steps is None or arr.shape[0] <= max_time_steps:
        return arr, 1

    stride = int(math.ceil(arr.shape[0] / max_time_steps))
    return arr[::stride], stride


def _add_horizon_marker(ax, horizon_time: float | None,
                        use_lyapunov: bool = False) -> None:
    """Draw the prediction-horizon marker with the contract's style and symbol."""
    if horizon_time is None:
        return
    hor = _ps().marker("prediction_horizon")
    ax.axvline(
        horizon_time,
        color=hor["color"],
        linestyle=hor["linestyle"],
        linewidth=hor["linewidth"],
        alpha=hor["alpha"],
        label=_ps().horizon_label(horizon_time, use_lyapunov),
    )


def plot_spatiotemporal_comparison(
    *,
    run_dir: Path,
    split: str,
    model_name: str,
    representation: str,
    output_base: Path,
    formats: list[str],
    dpi: int,
    max_time_steps: int | None = 1500,
    use_lyapunov: bool = False,
    lambda_max: float = 1.0,
    dt: float | None = None,
) -> list[str]:
    truth, pred = load_split_arrays(run_dir, split)
    metrics = load_split_metrics(run_dir, split)

    signed_error = truth - pred

    truth_plot, stride = _downsample_time(truth, max_time_steps)
    pred_plot, _ = _downsample_time(pred, max_time_steps)
    error_plot, _ = _downsample_time(signed_error, max_time_steps)

    dt = _resolve_dt(run_dir=run_dir, metrics=metrics, explicit_dt=dt)
    time_scale, x_label = _time_axis(use_lyapunov, lambda_max)

    n_time = truth.shape[0]
    extent = [0.0, (n_time - 1) * dt * time_scale, 0, truth.shape[1] - 1]

    horizon = _metric_value(metrics, "relative_l2_horizon_time")
    horizon_scaled = None if horizon is None else horizon * time_scale
    masked_from = _resolve_masked_from_index(metrics)

    # Panels (a)/(b) share one symmetric range derived from the DISPLAYED TRUE FIELD
    # ONLY; panel (c) gets its own symmetric error range. See common.plotting for why
    # the prediction must not influence the field range.
    vmin_state, vmax_state = _cp().truth_prediction_limits(truth_plot)
    vmin_error, vmax_error = _cp().error_limits(truth_plot, pred_plot)

    time_grid = np.arange(n_time) * dt * time_scale
    time_grid_plot = time_grid[::stride] if stride > 1 else time_grid

    pred_symbol = _pred_symbol(model_name, representation)

    with plt.rc_context(_ps().rc_params(_FAMILY_TRIPTYCH)):
        fig, axes = plt.subplots(
            3,
            1,
            figsize=_ps().figure_size_in(_FAMILY_TRIPTYCH),
            sharex=True,
            constrained_layout=True,
        )

        # a/b/c truth, prediction, difference -- the panel label is drawn inside
        # each axes, the title carries the mathematical field identifier only.
        # Colorbars carry no label: the panel identifier already names the field.
        panels = [
            (r"$u(x,t)$", truth_plot, vmin_state, vmax_state, "viridis"),
            (rf"${pred_symbol}(x,t)$", pred_plot, vmin_state, vmax_state, "viridis"),
            (
                rf"$u(x,t)-{pred_symbol}(x,t)$",
                error_plot,
                vmin_error,
                vmax_error,
                "RdBu_r",
            ),
        ]

        for ax, (identifier, data, vmin, vmax, cmap) in zip(axes, panels):
            image = ax.pcolormesh(
                time_grid_plot,
                _cp().kse_spatial_grid(data.shape[1]),
                data.T,
                cmap=cmap,
                shading="auto",
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            ax.set_ylabel(r"space $x$")
            # Same 5-unit spatial ticks as the AE-SINDy field triptychs; the axis
            # limits are left to the pcolormesh cell edges.
            ax.set_yticks(np.arange(0.0, _cp().KSE_DOMAIN_LENGTH, 5.0))
            ax.set_title(identifier, loc="left", x=0.35, pad=8)
            ax.grid(False)

            _add_horizon_marker(ax, horizon_scaled, use_lyapunov)

            if masked_from is not None:
                masked_time = int(masked_from) * dt * time_scale
                if 0 <= masked_time <= extent[1]:
                    cut = _ps().marker("safety_cutoff")
                    span = _ps().marker("masked_span")
                    ax.axvline(
                        masked_time,
                        color=cut["color"],
                        linestyle=cut["linestyle"],
                        linewidth=cut["linewidth"],
                        alpha=cut["alpha"],
                        label=_ps().cutoff_label(masked_time, use_lyapunov),
                    )
                    ax.axvspan(masked_time, extent[1], color=span["color"],
                               alpha=span["alpha"])

            fig.colorbar(image, ax=ax, pad=0.025, fraction=0.045)

        _cp().add_panel_labels(axes)

        # One marker legend for the whole triptych, on the error panel, matching the
        # AE-SINDy field-triptych convention. Every panel keeps its horizon and
        # safety-cutoff lines; only the duplicate legend boxes are dropped, so the
        # markers stay readable without three identical keys competing with the data.
        handles, _labels = axes[-1].get_legend_handles_labels()
        if handles:
            axes[-1].legend(
                loc=_ps().family(_FAMILY_TRIPTYCH)["legend_loc"], frameon=True)

        axes[-1].set_xlabel(x_label)

    return save_figure(fig, output_base, formats, dpi)
