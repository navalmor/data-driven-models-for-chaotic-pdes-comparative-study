"""Shared, pure plotting helpers for the reproducibility pipeline.

This module centralises *display-only* plotting decisions that were previously
duplicated across the simulation, AE-SINDy and forecasting plotting families:

* the ``time_axis_mode`` contract (``lyapunov`` / ``physical``) with its CLI +
  configuration precedence and its axis-label text;
* the display-time coordinate conversion (a pure multiply, never a mutation);
* the standalone-figure panel-label sequence ``(a)``, ``(b)``, ``(c)`` …;
* representation-aware forecasting model names and prediction symbols
  (``NGRC-64`` for full-state, ``NGRC-8`` for latent-space forecasts).

Design choice / location: this lives in the repository-root ``common`` package
so the three plotting families -- ``simulation``, ``ae_sindy`` and
``forecasting`` -- can all import one authority. Every entry point already puts
the repository root on ``sys.path`` (the same bootstrap ``common.logging_setup``
relies on), so no *library-level* ``sys.path`` mutation is introduced here.

The module is import-safe: it defines constants and pure functions only. It
performs **no file I/O, consumes no randomness, mutates no arrays, and changes
no scientific data**. A change of ``time_axis_mode`` only rescales the displayed
x-coordinate and relabels the axis; it never changes which samples are shown or
how many.
"""

from __future__ import annotations

import argparse
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Time-axis mode contract
# ---------------------------------------------------------------------------

TIME_AXIS_MODES: tuple[str, ...] = ("lyapunov", "physical")
DEFAULT_TIME_AXIS_MODE: str = "lyapunov"

# mathtext-compatible (no external LaTeX required)
LYAPUNOV_TIME_LABEL: str = r"time $t$ [$1/\lambda_{\max}$]"
PHYSICAL_TIME_LABEL: str = r"time $t$"


def _normalise_mode(value: Any) -> str:
    mode = str(value).strip().lower()
    if mode not in TIME_AXIS_MODES:
        raise ValueError(
            f"time_axis_mode must be one of {TIME_AXIS_MODES}, got {value!r}."
        )
    return mode


def resolve_time_axis_mode(
    cli_value: Optional[str] = None,
    config_value: Optional[str] = None,
    default: str = DEFAULT_TIME_AXIS_MODE,
) -> str:
    """Resolve the display-time mode with precedence CLI > config > default.

    The first non-``None`` of ``cli_value`` / ``config_value`` / ``default`` is
    selected and validated; an unsupported value raises ``ValueError``. A value
    that is not selected (e.g. a bad config overridden by a valid CLI flag) is
    left untouched, matching the documented precedence.
    """
    for candidate in (cli_value, config_value, default):
        if candidate is None:
            continue
        return _normalise_mode(candidate)
    # default is never None in practice; guard anyway.
    return _normalise_mode(default)


def time_axis_scale_and_label(
    mode: str,
    *,
    dt: Optional[float] = None,
    lambda_max: Optional[float] = None,
) -> tuple[float, str]:
    """Return ``(time_scale, axis_label)`` for a resolved display mode.

    * ``physical``  -> scale ``1.0``, label ``time $t$``.
    * ``lyapunov`` -> scale ``lambda_max``, label ``time $t$ [$1/\\lambda_{\\max}$]``.

    ``dt`` (when supplied) must be positive; ``lambda_max`` must be positive for
    Lyapunov mode. No authoritative scientific constant is invented here -- the
    caller passes the existing ``dt`` / ``lambda_max``.
    """
    mode = _normalise_mode(mode)

    if dt is not None and float(dt) <= 0.0:
        raise ValueError(f"dt must be positive, got {dt!r}.")

    if mode == "lyapunov":
        if lambda_max is None or float(lambda_max) <= 0.0:
            raise ValueError(
                f"lyapunov time_axis_mode requires a positive lambda_max, got {lambda_max!r}."
            )
        return float(lambda_max), LYAPUNOV_TIME_LABEL

    # physical
    return 1.0, PHYSICAL_TIME_LABEL


def scale_time(times: Any, time_scale: float) -> np.ndarray:
    """Return ``times * time_scale`` as a new float array (input never mutated)."""
    arr = np.asarray(times, dtype=float)
    # np multiplication always returns a new array, so the caller's array is safe.
    return arr * float(time_scale)


def add_time_axis_argument(parser: argparse.ArgumentParser) -> None:
    """Register ``--time-axis-mode {lyapunov,physical}`` on a plotting CLI.

    Default is ``None`` so that configuration resolution can distinguish "not
    given on the CLI" from an explicit choice. Display-only: it changes the
    x-coordinate scaling and label, never the underlying samples.
    """
    parser.add_argument(
        "--time-axis-mode",
        choices=list(TIME_AXIS_MODES),
        default=None,
        help=(
            "Display-time coordinate for time-based plots: 'lyapunov' shows "
            "time in Lyapunov units (t*lambda_max), 'physical' shows physical "
            "simulation time. Display-only; does not change the samples shown. "
            "Precedence: this flag, then plotting.time_axis_mode in the config, "
            "then 'lyapunov'."
        ),
    )


def _config_time_axis_mode(config: Any) -> Optional[str]:
    if isinstance(config, dict):
        plotting = config.get("plotting")
        if isinstance(plotting, dict):
            value = plotting.get("time_axis_mode")
            if value is not None:
                return str(value)
    return None


def resolve_time_axis_mode_from(
    args: Any = None,
    config: Any = None,
    default: str = DEFAULT_TIME_AXIS_MODE,
) -> str:
    """Resolve the mode from an argparse namespace and/or a loaded config dict.

    Reads ``args.time_axis_mode`` (CLI) and ``config['plotting']['time_axis_mode']``
    and applies precedence CLI > config > default. Missing ``plotting`` section
    or missing field falls through to the default -- existing configs need no
    new field.
    """
    cli_value = getattr(args, "time_axis_mode", None) if args is not None else None
    return resolve_time_axis_mode(
        cli_value=cli_value,
        config_value=_config_time_axis_mode(config),
        default=default,
    )


# ---------------------------------------------------------------------------
# Physical spatial coordinate
# ---------------------------------------------------------------------------

# Domain length of the canonical KSE simulation (configs/00_simulation: L=22, nx=64).
KSE_DOMAIN_LENGTH: float = 22.0


def kse_spatial_grid(n_points: int, domain_length: float = KSE_DOMAIN_LENGTH) -> np.ndarray:
    """Physical spatial coordinates ``x_j = j * domain_length / n_points`` on ``[0, L)``.

    The same grid the solver integrates on (``np.linspace(0, L, nx, endpoint=False)``),
    so a field panel is drawn against the physical coordinate instead of the grid index.
    Display-only: it relabels the spatial axis and changes no sample.
    """
    n_points = int(n_points)
    domain_length = float(domain_length)
    if n_points <= 0:
        raise ValueError(f"n_points must be positive, got {n_points!r}.")
    if domain_length <= 0.0:
        raise ValueError(f"domain_length must be positive, got {domain_length!r}.")
    return np.linspace(0.0, domain_length, n_points, endpoint=False)


# ---------------------------------------------------------------------------
# Panel labels
# ---------------------------------------------------------------------------

def panel_label(index: int) -> str:
    """Standalone-figure panel label for a zero-based ``index``: 0 -> '(a)'."""
    if index < 0:
        raise ValueError(f"panel index must be >= 0, got {index}.")
    if index >= 26:
        raise ValueError(f"panel index {index} exceeds the supported (a)-(z) range.")
    return f"({chr(ord('a') + index)})"


def panel_title(index: int, identifier: str) -> str:
    """Compose ``'(a) <identifier>'`` keeping the label attached to its identifier."""
    return f"{panel_label(index)} {identifier}"


def panel_letter(index: int) -> str:
    """Standalone lowercase panel letter for a zero-based ``index``: 0 -> 'a'.

    Unlike :func:`panel_label` this returns the bare letter with no parentheses,
    for the in-axes panel-label convention drawn by :func:`add_panel_label`.
    """
    if index < 0:
        raise ValueError(f"panel index must be >= 0, got {index}.")
    if index >= 26:
        raise ValueError(f"panel index {index} exceeds the supported a-z range.")
    return chr(ord("a") + index)


# Placement profiles for :func:`add_panel_label`.
#
# The label sits at ``x`` from the left spine and ``1 - y`` from the top spine
# in *axes* fractions, so the two insets only look equal when
# ``(1 - y) * axes_height == x * axes_width``. A wide field panel and a
# near-square line panel therefore need a different ``y`` to read the same. The
# profiles keep those few family-specific coordinates here, next to the style,
# instead of scattering literals through the scientific plotting modules.
PANEL_LABEL_PROFILES: dict[str, dict[str, Any]] = {
    # Wide field/latent triptych panels (aspect ~2.4).
    # The corner sits over a non-salient edge of the field, so upper-left is free.
    "field_triptych": {"x": 0.02, "y": 0.95, "va": "top"},
    # Near-square line panels (aspect ~1.3), e.g. training-history curves.
    # Documented corner exception: on a decaying loss/step curve the data
    # *maximum* is the first sample, so the upper-left corner is occupied by the
    # very quantity the panel is about and a label there would hide it. The
    # legend sits upper-right, and the region under a decaying curve is empty,
    # so the label goes lower-left -- the only corner free of both.
    "descending_line_panel": {"x": 0.02, "y": 0.03, "va": "bottom"},
}

DEFAULT_PANEL_LABEL_PROFILE: str = "field_triptych"


def panel_label_placement(profile: str = DEFAULT_PANEL_LABEL_PROFILE) -> dict[str, Any]:
    """Return the ``{'x': …, 'y': …, 'va': …}`` axes-fraction placement for a profile."""
    if profile not in PANEL_LABEL_PROFILES:
        raise ValueError(
            f"panel-label profile must be one of {sorted(PANEL_LABEL_PROFILES)}, got {profile!r}."
        )
    return dict(PANEL_LABEL_PROFILES[profile])


def add_panel_label(
    ax: Any,
    label: str,
    *,
    profile: str = DEFAULT_PANEL_LABEL_PROFILE,
    x: Optional[float] = None,
    y: Optional[float] = None,
    fontsize: float = 13.0,
    fontweight: str = "normal",
    color: str = "black",
    zorder: float = 10.0,
    pad: float = 0.2,
    facecolor: str = "white",
    edgecolor: str = "0.6",
    linewidth: float = 0.5,
    alpha: float = 0.9,
    va: Optional[str] = None,
) -> Any:
    """Draw a standalone in-panel label (e.g. ``'a'``) inside an axes' corner.

    Display-only. The label is placed in *axes* coordinates
    (``transform=ax.transAxes``) on top of the plotted data (high ``zorder``)
    inside a small legend-like background box so it stays legible over both
    bright and dark regions. The label is a light
    ``normal``-weight letter in a thin-bordered white box. Placement comes from
    ``profile`` (see :data:`PANEL_LABEL_PROFILES`); explicit ``x``/``y``/``va``
    override it for a one-off panel. It adds one text artist and nothing else:
    no file I/O, no randomness, no array mutation, no scientific change. No
    plotting backend is imported here -- the caller supplies ``ax`` -- so the
    module keeps its import-safe, pure character.

    Returns the created ``matplotlib.text.Text`` artist.
    """
    placement = panel_label_placement(profile)
    x = placement["x"] if x is None else float(x)
    y = placement["y"] if y is None else float(y)
    va = placement.get("va", "top") if va is None else str(va)

    return ax.text(
        x,
        y,
        str(label),
        transform=ax.transAxes,
        ha="left",
        va=va,
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
        zorder=zorder,
        bbox={
            "boxstyle": f"round,pad={pad}",
            "facecolor": facecolor,
            "edgecolor": edgecolor,
            "linewidth": linewidth,
            "alpha": alpha,
        },
    )


def add_panel_labels(
    axes: Any,
    *,
    profile: str = DEFAULT_PANEL_LABEL_PROFILE,
    start: int = 0,
    **kwargs: Any,
) -> list:
    """Label a sequence of axes ``a``, ``b``, ``c`` … in order.

    The multi-panel convention in one call: every standalone multi-panel figure
    labels its axes this way, so the letter sequence is generated once here
    rather than re-derived at each call site. ``start`` offsets the first letter;
    remaining keyword arguments are forwarded to :func:`add_panel_label`.
    """
    return [
        add_panel_label(ax, panel_letter(start + offset), profile=profile, **kwargs)
        for offset, ax in enumerate(axes)
    ]


# ---------------------------------------------------------------------------
# Forecasting model naming (representation-aware)
# ---------------------------------------------------------------------------

_REPRESENTATION_DIM = {
    "full64": "64",
    "latent8": "8",
}


def representation_dim(representation: str) -> str:
    """Map a representation slug to its forecasting suffix ('full64' -> '64')."""
    key = str(representation).strip().lower()
    if key not in _REPRESENTATION_DIM:
        raise ValueError(
            f"representation must be one of {sorted(_REPRESENTATION_DIM)}, got {representation!r}."
        )
    return _REPRESENTATION_DIM[key]


def model_display_name(model_name: str, representation: str) -> str:
    """Human display name for a forecasting model, e.g. ('NGRC','latent8') -> 'NGRC-8'."""
    return f"{model_name}-{representation_dim(representation)}"


def prediction_symbol(model_name: str, representation: str) -> str:
    r"""Mathtext prediction symbol, e.g. 'NGRC'+'latent8' -> ``\hat{u}_{\mathrm{NGRC-8}}``.

    ``full64`` yields the ``-64`` suffix and ``latent8`` the ``-8`` suffix, so a
    latent-space forecast is never mislabelled ``-64``.
    """
    return rf"\hat{{u}}_{{\mathrm{{{model_display_name(model_name, representation)}}}}}"


# ---------------------------------------------------------------------------
# Rollout-triptych colour scales
# ---------------------------------------------------------------------------
#
# Every true / prediction / error three-panel figure in the repository shares one
# convention, so panels are comparable across models, representations and splits:
#
#   panels (a) and (b)  one symmetric range derived from the DISPLAYED TRUE FIELD ONLY
#   panel  (c)          an independent symmetric range derived from the displayed error
#
# Deriving the field range from the truth alone is the whole point. A diverging
# prediction must not be allowed to inflate the shared range, because that flattens the
# true field into a featureless band and hides the very behaviour the figure exists to
# show. When a prediction leaves the truth's amplitude range it saturates, and that
# saturation is the scientifically informative signal -- it is never suppressed by
# rescaling. Only the *display* normalisation saturates; the underlying arrays are never
# clipped, smoothed or otherwise modified.

# Smallest non-degenerate half-range, used when a finite field is identically zero so a
# singular normalisation (vmin == vmax) is never constructed.
MIN_COLOR_HALF_RANGE: float = 1e-12


def _symmetric_limit(values: Any, *, what: str) -> float:
    """Largest finite absolute value of ``values``, floored away from zero.

    Raises ``ValueError`` if nothing is finite -- a silent fallback would produce a
    plausible-looking figure from unusable data.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError(
            f"cannot derive colour limits: the displayed {what} contains no finite values."
        )
    return max(float(np.max(np.abs(finite))), MIN_COLOR_HALF_RANGE)


def truth_prediction_limits(u_true_displayed: Any) -> tuple[float, float]:
    """Shared ``(vmin, vmax)`` for the true and predicted field panels.

    Derived from the displayed true field alone::

        field_limit = nanmax(abs(u_true_displayed))
        (vmin, vmax) = (-field_limit, +field_limit)

    The prediction is deliberately not consulted, so prediction outliers cannot widen
    the scale. No percentile is used: a percentile would silently clip the truth's own
    extremes.
    """
    limit = _symmetric_limit(u_true_displayed, what="true field")
    return -limit, limit


def error_limits(u_true_displayed: Any, u_pred_displayed: Any) -> tuple[float, float]:
    """Independent zero-centred ``(vmin, vmax)`` for the error panel.

    Derived from the displayed signed error alone::

        error_limit = nanmax(abs(u_true_displayed - u_pred_displayed))
        (vmin, vmax) = (-error_limit, +error_limit)

    Independent of the field limits by design: reusing them would hide large errors
    whenever the error exceeds the truth's amplitude.
    """
    true_arr = np.asarray(u_true_displayed, dtype=float)
    pred_arr = np.asarray(u_pred_displayed, dtype=float)
    limit = _symmetric_limit(true_arr - pred_arr, what="error field")
    return -limit, limit


def prediction_saturation(u_true_displayed: Any, u_pred_displayed: Any) -> dict[str, float]:
    """Audit how far the prediction leaves the truth-derived display range.

    Returns the field limit, the prediction extremes, and the fraction of finite
    predicted values falling outside the shared range. Saturation is reported, never
    corrected.
    """
    _, field_limit = truth_prediction_limits(u_true_displayed)
    pred = np.asarray(u_pred_displayed, dtype=float)
    finite = pred[np.isfinite(pred)]
    n_finite = int(finite.size)
    n_saturated = int(np.count_nonzero(np.abs(finite) > field_limit))
    return {
        "field_limit": field_limit,
        "prediction_min": float(np.min(finite)) if n_finite else float("nan"),
        "prediction_max": float(np.max(finite)) if n_finite else float("nan"),
        "n_finite_prediction": n_finite,
        "n_saturated": n_saturated,
        "saturation_fraction": (n_saturated / n_finite) if n_finite else 0.0,
        "n_nonfinite_prediction": int(pred.size - n_finite),
    }
