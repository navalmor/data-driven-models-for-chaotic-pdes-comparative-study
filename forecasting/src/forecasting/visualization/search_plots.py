from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from forecasting.visualization.style import save_figure


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    label: str
    kind: str = "continuous"
    scale: str = "linear"
    include_in_matrix: bool = True
    include_in_surrogate: bool = True


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False

    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _to_float(value: Any, *, scale: str) -> float:
    if value is None or str(value).strip() == "":
        return float("nan")

    x = float(value)

    if scale == "log10":
        if x <= 0:
            return float("nan")
        return math.log10(x)

    if scale != "linear":
        raise ValueError(f"Unsupported parameter scale: {scale!r}")

    return x


def _unique_nonempty(rows: list[dict[str, Any]], name: str) -> list[str]:
    return sorted(
        {
            str(row.get(name, "")).strip()
            for row in rows
            if str(row.get(name, "")).strip() != ""
        }
    )


def _parameter_values(rows: list[dict[str, Any]], spec: ParameterSpec) -> tuple[np.ndarray, dict[int, str] | None]:
    if spec.kind == "boolean":
        return (
            np.asarray([1.0 if _as_bool(row.get(spec.name)) else 0.0 for row in rows], dtype=float),
            {0: "false", 1: "true"},
        )

    if spec.kind == "categorical":
        categories = _unique_nonempty(rows, spec.name)
        mapping = {cat: i for i, cat in enumerate(categories)}
        values = np.asarray(
            [float(mapping[str(row.get(spec.name, "")).strip()]) for row in rows],
            dtype=float,
        )
        return values, {i: cat for cat, i in mapping.items()}

    return (
        np.asarray([_to_float(row.get(spec.name), scale=spec.scale) for row in rows], dtype=float),
        None,
    )


def build_parameter_specs(raw_specs: list[dict[str, Any]]) -> list[ParameterSpec]:
    specs: list[ParameterSpec] = []

    for raw in raw_specs:
        name = str(raw["name"])
        specs.append(
            ParameterSpec(
                name=name,
                label=str(raw.get("label", name)),
                kind=str(raw.get("kind", "continuous")),
                scale=str(raw.get("scale", "linear")),
                include_in_matrix=bool(raw.get("include_in_matrix", True)),
                include_in_surrogate=bool(raw.get("include_in_surrogate", True)),
            )
        )

    return specs


def varied_specs_for_matrix(rows: list[dict[str, Any]], specs: list[ParameterSpec]) -> list[ParameterSpec]:
    out: list[ParameterSpec] = []

    for spec in specs:
        if not spec.include_in_matrix:
            continue
        if len(_unique_nonempty(rows, spec.name)) <= 1:
            continue
        out.append(spec)

    return out


def varied_specs_for_surrogate(rows: list[dict[str, Any]], specs: list[ParameterSpec]) -> list[ParameterSpec]:
    out: list[ParameterSpec] = []

    for spec in specs:
        if not spec.include_in_surrogate:
            continue
        if spec.kind == "categorical":
            continue
        if len(_unique_nonempty(rows, spec.name)) <= 1:
            continue
        out.append(spec)

    return out


def ok_rows(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for row in rows:
        if str(row.get("status", "ok")) != "ok":
            continue

        try:
            value = float(row[metric])
        except Exception:
            continue

        if math.isfinite(value):
            out.append(row)

    if not out:
        raise ValueError(f"No successful finite rows found for metric={metric!r}.")

    return out


def _objective_values(rows: list[dict[str, Any]], metric: str) -> np.ndarray:
    return np.asarray([float(row[metric]) for row in rows], dtype=float)


def _trial_values(rows: list[dict[str, Any]]) -> np.ndarray:
    values = []

    for i, row in enumerate(rows, start=1):
        try:
            values.append(int(row.get("trial", i)))
        except Exception:
            values.append(i)

    return np.asarray(values, dtype=int)


def _best_row(rows: list[dict[str, Any]], metric: str, objective: str) -> dict[str, Any]:
    reverse = objective == "maximize"
    return sorted(rows, key=lambda row: float(row[metric]), reverse=reverse)[0]


# Display names for the skopt optimizer families, verified against
# forecasting/scripts/optimizer_search_ngrc.py::_optimizer_for_name:
# "dummy" has no surrogate (pure random search), "forest"/"gbrt"/"gp" map to
# skopt base_estimator "RF"/"GBRT"/"GP" respectively.
_OPTIMIZER_DISPLAY_NAMES = {
    "dummy": "Random",
    "forest": "Random forest",
    "gbrt": "GBRT",
    "gp": "Gaussian process",
}

# Nicer notation for the search-space parameter labels configured in
# search_plots.parameters[].label, shared by the sampling-matrix and
# surrogate-landscape plots.
# Lowercase sentence case (CH-07). The plot configs already store these labels
# in lowercase; the table previously re-capitalised five of the six, which is
# why "quadratic mode" -- absent from it -- rendered differently from its
# neighbours on the same matrix. Only the maths label needs an override now.
def _ps():
    """Lazy accessor for the central style contract (see ``_cp``).

    ``common`` is imported inside the call, not at module import time, so this
    library package never assumes the repository root is on ``sys.path``.
    """
    from common import plot_style

    return plot_style


_FAMILY_CONVERGENCE = "optimizer_convergence"
_FAMILY_MATRIX = "optimizer_sampling_matrix"
_FAMILY_SURROGATE = "optimizer_surrogate_landscape"


def _matrix_figsize(n: int) -> tuple[float, float]:
    """Class-B canvas for an n x n sampling matrix, at printed size.

    The matrix is printed on a rotated page whose usable width is the text
    HEIGHT. Drawing at the historical 2.75*n+1.2 inches produced a 16.8 in
    canvas that LaTeX shrank to 7.6 in, so a 10 pt tick label reached paper at
    4.5 pt. The proportions are preserved; only the scale changes.
    """
    aspect = (2.55 * n + 1.1) / (2.75 * n + 1.2)
    return _ps().figure_size_in(_FAMILY_MATRIX, aspect_ratio=aspect)


def _surrogate_figsize(n: int) -> tuple[float, float]:
    aspect = (2.65 * n + 1.15) / (2.85 * n + 1.35)
    return _ps().figure_size_in(_FAMILY_SURROGATE, aspect_ratio=aspect)


_PARAM_LABEL_OVERRIDES = {
    "log10 ridge alpha": r"$\log_{10}(\alpha_{\mathrm{ridge}})$",
}

# Display-only mappings for categorical / boolean parameter *values*. The stored
# optimiser records and the selected candidate are never changed; only the
# rendered tick text is remapped. "full" is the complete NVAR feature library
# (linear + quadratic + sin + cos, confirmed from
# optimizer_search_ngrc.py::_feature_flags), hence "Full library".
_CATEGORICAL_VALUE_DISPLAY = {
    "linear_quadratic": "Linear + quadratic",
    "full": "Full library",
    "true": "Yes",
    "false": "No",
}

# Wrapped display forms for the long feature-set names, used on *both* axes so
# the two categories read identically wherever they appear. On an x-axis the
# single-line forms were wide enough to need rotating, which read unevenly and
# competed with the "Feature set" axis label; on a y-axis they crowded the left
# margin. Two upright lines fix both at full font size. Only the *rendered* text
# differs: the stored value, the category ordering and the selected candidate
# are untouched. Values absent here (booleans "Yes"/"No") are already short
# enough to render on one line and fall through to the single-line mapping.
_CATEGORICAL_VALUE_DISPLAY_WRAPPED = {
    "linear_quadratic": "Linear +\nquadratic",
    "full": "Full\nlibrary",
}

# Categorical tick/axis-label spacing, in points. Two-line x-tick labels need a
# larger gap to the axis label than single-line ones so the second line never
# touches the "Feature set" text; both values are deliberate and nonzero.
_CATEGORICAL_TICK_PAD = 4.0
_XLABEL_PAD_SINGLE_LINE = 6.0
_XLABEL_PAD_MULTILINE = 12.0

# Sampling-matrix figure margins. Bottom and top are named so the colorbar axes
# can be anchored to exactly the same band as the panel matrix. The left margin
# holds the categorical y-tick labels; wrapping them onto two lines roughly
# halves their width ("quadratic" instead of "Linear + quadratic"), so it is
# reduced accordingly rather than left oversized.
_MATRIX_LEFT = 0.105
_MATRIX_BOTTOM = 0.125
_MATRIX_TOP = 0.96

# Colorbar tick formatting. Matplotlib's default ScalarFormatter switches to a
# detached offset ("1e-10 + 1.496544027e1") once the tick spacing is tiny
# relative to the tick magnitude. That notation is unreadable in print and, for
# a surrogate panel whose prediction is effectively constant, actively
# misleading: it renders numerical noise as if it were structure.
#
# _COLORBAR_MAX_DECIMALS caps how many fixed-point decimals an ordinary
# (non-degenerate) label may use before the range is treated as degenerate; six
# is the most a thesis colorbar can carry without adjacent labels colliding.
# _COLORBAR_DEGENERATE_REL_SPAN is the relative span below which a colorbar is
# considered constant to displayable precision. 1e-8 sits far above float64
# round-off and far below any real variation in a validation horizon.
# _COLORBAR_DEGENERATE_MAX_DECIMALS bounds the search for a precision that keeps
# the three degenerate-range ticks distinct: 12 stays within float64's ~13
# honest decimals for a value near 15, so a label is never fabricated noise.
_COLORBAR_MAX_DECIMALS = 6
_COLORBAR_DEGENERATE_REL_SPAN = 1e-8
_COLORBAR_DEGENERATE_MAX_DECIMALS = 12


def _tick_decimals(step: float) -> int:
    """Fixed-point decimals needed to distinguish ticks spaced ``step`` apart.

    Returns 0 for integer-ish spacing, so ordinary colorbars keep the concise
    labels they already had ("160", "180", ...).
    """
    if not math.isfinite(step) or step <= 0.0:
        return 0

    return max(0, -math.floor(math.log10(step)))


def _concise_value(value: float, decimals: int) -> str:
    """Fixed-point text with trailing zeros trimmed, and no negative zero."""
    text = f"{value:.{decimals}f}"

    if "." in text:
        text = text.rstrip("0").rstrip(".")

    return text if set(text) - {"-", "0"} else text.lstrip("-")


def _distinct_fixed_decimals(values: list[float], *, max_decimals: int) -> int | None:
    """Smallest fixed-point precision that renders ``values`` as distinct labels.

    Returns the number of decimals in ``[0, max_decimals]`` at which every value
    formats to a different string, or ``None`` if even ``max_decimals`` cannot
    separate them (only possible when two values are equal to that precision).
    Fixed precision is deliberate: all labels then share one decimal count, so a
    column of tick values reads consistently instead of ragged.
    """
    for decimals in range(0, max_decimals + 1):
        labels = [f"{value:.{decimals}f}" for value in values]
        if len(set(labels)) == len(labels):
            return decimals

    return None


def _apply_colorbar_tick_format(colorbar: Any) -> dict[str, Any]:
    """Replace automatic offset notation with explicit fixed-point tick labels.

    Policy, in order:

    1. Disable the automatic offset unconditionally.
    2. Derive the decimals needed to distinguish adjacent ticks and use ordinary
       fixed-point labels. Normal ranges need zero decimals and are unchanged.
    3. If the required precision exceeds ``_COLORBAR_MAX_DECIMALS``, or the span
       is below ``_COLORBAR_DEGENERATE_REL_SPAN`` of the magnitude, the mapped
       surface is constant to ordinary precision. Rather than collapse it to one
       tick, place three ticks at ``vmin``, the midpoint and ``vmax`` and format
       all three as complete absolute values at the least precision that keeps
       them distinct. The reader then sees the true horizon value and that the
       surface is flat, without any detached offset.

    This is display-only: colour limits, normalization and the mapped array are
    never touched. Returns the decision for logging and tests.
    """
    from matplotlib.ticker import FuncFormatter

    vmin, vmax = (float(v) for v in colorbar.mappable.get_clim())
    span = abs(vmax - vmin)
    magnitude = max(abs(vmin), abs(vmax))

    ticks = [float(t) for t in np.asarray(colorbar.get_ticks(), dtype=float) if math.isfinite(t)]
    steps = [b - a for a, b in zip(ticks, ticks[1:]) if b > a]
    step = min(steps) if steps else span
    decimals = _tick_decimals(step)

    degenerate = (
        span <= 0.0
        or span <= _COLORBAR_DEGENERATE_REL_SPAN * magnitude
        or decimals > _COLORBAR_MAX_DECIMALS
    )

    tick_decimals: int | None = None

    if degenerate:
        midpoint = 0.5 * (vmin + vmax)
        tick_values = [vmin, midpoint, vmax]
        tick_decimals = _distinct_fixed_decimals(
            tick_values, max_decimals=_COLORBAR_DEGENERATE_MAX_DECIMALS
        )

        if tick_decimals is None:
            # Only reachable when vmin and vmax are equal to 12 decimals, i.e. a
            # genuinely constant surface where no absolute labels can differ. A
            # single honest tick is then the only truthful presentation.
            label = _concise_value(midpoint, _COLORBAR_MAX_DECIMALS)
            colorbar.set_ticks([midpoint], labels=[label])
        else:
            labels = [f"{value:.{tick_decimals}f}" for value in tick_values]
            colorbar.set_ticks(tick_values, labels=labels)
    else:
        colorbar.formatter = FuncFormatter(
            lambda value, _pos, _d=decimals: _concise_value(value, _d)
        )
        colorbar.update_ticks()

    colorbar.ax.yaxis.set_offset_position("left")
    colorbar.ax.yaxis.get_offset_text().set_visible(False)

    return {
        "degenerate": degenerate,
        "decimals": decimals,
        "tick_decimals": tick_decimals,
        "span": span,
        "step": step,
    }


# A matrix cell is roughly 1.3 in wide at printed size, which holds about
# fourteen characters at the dense type scale. Longer parameter names are
# wrapped rather than shortened or scaled: the exact wording is what the thesis
# text uses, and adjacent panels' labels collided when they ran past their own
# cell. Widening the panel spacing does not help, because a centred label grows
# outward from its own centre no matter how far apart the panels are.
_PARAM_LABEL_WRAP_CHARS = 14


def _wrap_param_label(label: str) -> str:
    if "$" in label or len(label) <= _PARAM_LABEL_WRAP_CHARS:
        return label
    words = label.split()
    if len(words) < 2:
        return label
    best, best_cost = None, None
    for cut in range(1, len(words)):
        head, tail = " ".join(words[:cut]), " ".join(words[cut:])
        cost = max(len(head), len(tail))
        if best_cost is None or cost < best_cost:
            best, best_cost = f"{head}\n{tail}", cost
    return best


def _fold_offset_into_label(ax, *, axis: str = "x") -> None:
    """Move a scientific-notation exponent from the offset text into the label.

    Matplotlib parks the shared exponent at the far end of an axis, immediately
    above a centred axis label. On a matrix cell about an inch wide the two land
    on each other -- "lambda physics" rendered as "lambda ph1e-6ics".

    The exponent is appended on its own line rather than inline, so the label
    gains height instead of width: a wider label would simply move the collision
    to the neighbouring panel. The offset artist is then hidden, which is what
    actually removes it from the drawing.
    """
    if axis not in {"x", "y"}:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    target = ax.xaxis if axis == "x" else ax.yaxis
    offset_artist = target.get_offset_text()
    offset = offset_artist.get_text().strip()
    label = target.get_label().get_text()
    if not offset or not label or "\\times" in label:
        return
    # matplotlib writes a Unicode minus (U+2212), not an ASCII hyphen
    m = re.fullmatch(r"1e([+\u2212-]?)(\d+)", offset)
    if m:
        sign = "-" if m.group(1) in {"-", "\u2212"} else ""
        pretty = rf"$\times 10^{{{sign}{int(m.group(2))}}}$"
    else:
        pretty = offset.replace("\u2212", "-")
    offset_artist.set_visible(False)
    target.set_label_text(f"{label}\n{pretty}")


def _fold_offsets_for_figure(fig) -> None:
    """Tidy every axis exponent in a panel matrix, once the figure is drawn.

    Three cases, and they need different treatment:

      additive offset      "+2.718e7" means the ticks are shown relative to a
                           base. Hiding it would make them unreadable, so it is
                           suppressed at source instead and the ticks carry
                           their full values.
      labelled axis        the shared multiplicative exponent is folded into the
                           label, on its own line so the label grows downward
                           rather than sideways into the next panel.
      unlabelled axis      interior panels of a scatter matrix share the column's
                           variable and scale, so the exponent is already shown
                           on the labelled panel below. The duplicate is hidden.

    The offset text does not exist until the first draw, so none of this can be
    done where the labels are set.
    """
    for ax in fig.axes:
        for name in ("x", "y"):
            try:
                ax.ticklabel_format(axis=name, useOffset=False)
            except (AttributeError, ValueError):
                pass                      # categorical or log axis: nothing to do
    def _apply(_event=None):
        for ax in fig.axes:
            for name in ("x", "y"):
                target = ax.xaxis if name == "x" else ax.yaxis
                if target.get_label().get_text().strip():
                    _fold_offset_into_label(ax, axis=name)
                else:
                    target.get_offset_text().set_visible(False)

    fig.canvas.draw()
    _apply()
    # The formatter regenerates the offset text on every draw, so hiding it once
    # is undone by the draw that savefig performs. Re-applying on each draw is
    # what makes it stick.
    fig.canvas.mpl_connect("draw_event", _apply)


def _display_param_label(spec: "ParameterSpec", *, axis: str = "x") -> str:
    """Display form of a parameter label, wrapped only where wrapping helps.

    Both axes wrap, for different reasons. An x-axis label grows sideways under
    its own cell and runs into the neighbouring panel. A rotated y-axis label
    grows ALONG its panel and, on a matrix row roughly 0.9 in tall, an unwrapped
    name is taller than the row and collides with the label of the row above.
    Wrapping costs a y-label one line-height of horizontal room, which is why
    the left margin is set wide enough to absorb it.

    ``axis`` is retained and validated because the two axes are laid out by
    different call sites and a future divergence should be explicit.
    """
    if axis not in {"x", "y"}:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    label = _PARAM_LABEL_OVERRIDES.get(spec.label, spec.label)
    return _wrap_param_label(label)


def _display_category(value: Any, *, axis: str = "y") -> str:
    """Map a raw categorical/boolean tick value to its academic display form.

    The long feature-set names are wrapped onto two lines on both axes, so the
    same category reads identically wherever it appears. ``axis`` is retained
    and validated because the two axes still differ in *formatting* (alignment
    and padding are chosen per axis by the callers below). Display text only:
    the stored value is never rewritten.
    """
    if axis not in {"x", "y"}:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    key = str(value).strip().lower()

    if key in _CATEGORICAL_VALUE_DISPLAY_WRAPPED:
        return _CATEGORICAL_VALUE_DISPLAY_WRAPPED[key]

    return _CATEGORICAL_VALUE_DISPLAY.get(key, str(value))


def _is_multiline_category(ticks: dict[int, str] | None, *, axis: str) -> bool:
    """True when any tick label for this parameter wraps to two lines."""
    if not ticks:
        return False
    return any("\n" in _display_category(v, axis=axis) for v in ticks.values())


def _lyapunov_time_axis(use_lyapunov: bool, lambda_max: float) -> float:
    return float(lambda_max) if use_lyapunov else 1.0


def plot_optimizer_convergence(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    objective: str,
    output_base: Path,
    formats: list[str],
    dpi: int,
    use_lyapunov: bool = False,
    lambda_max: float = 1.0,
) -> list[str]:
    # Type sizes come from the contract, not from this module.
    with plt.rc_context(_ps().rc_params(_FAMILY_CONVERGENCE)):
        rows = ok_rows(rows, metric)

        time_scale = _lyapunov_time_axis(use_lyapunov, lambda_max)
        y_label = (
            rf"{_ps().term('best_so_far_validation_horizon')}, $T_h$ [$1/\lambda_{{\max}}$]"
            if use_lyapunov
            else rf"{_ps().term('best_so_far_validation_horizon')}, $T_h$"
        )

        fig, ax = plt.subplots(figsize=_ps().figure_size_in(_FAMILY_CONVERGENCE),
                               constrained_layout=True)

        for optimizer in sorted({str(row["optimizer"]) for row in rows}):
            opt_rows = [row for row in rows if str(row["optimizer"]) == optimizer]
            opt_rows = sorted(opt_rows, key=lambda row: int(row.get("trial", 0) or 0))

            x = _trial_values(opt_rows)
            y = _objective_values(opt_rows, metric)

            if objective == "maximize":
                best_so_far = np.maximum.accumulate(y)
            elif objective == "minimize":
                best_so_far = np.minimum.accumulate(y)
            else:
                raise ValueError(f"objective must be maximize/minimize, got {objective!r}")

            best_so_far = best_so_far * time_scale
            display_name = _OPTIMIZER_DISPLAY_NAMES.get(optimizer, optimizer)

            ax.plot(x, best_so_far, marker="o", markersize=3.0, linewidth=1.8, label=display_name)

        ax.set_xlabel(_ps().axis_label("Optimizer trial"))
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
        ax.legend(title=_ps().axis_label("Optimizer"),
                  loc=_ps().family(_FAMILY_CONVERGENCE)["legend_loc"])

        return save_figure(fig, output_base, formats, dpi)


def plot_sampling_matrix(
    rows: list[dict[str, Any]],
    *,
    specs: list[ParameterSpec],
    metric: str,
    objective: str,
    output_base: Path,
    formats: list[str],
    dpi: int,
    use_lyapunov: bool = False,
    lambda_max: float = 1.0,
) -> list[str]:
    # Type sizes come from the contract, not from this module.
    with plt.rc_context(_ps().rc_params(_FAMILY_MATRIX)):
        rows = ok_rows(rows, metric)
        specs = varied_specs_for_matrix(rows, specs)

        if not specs:
            raise ValueError("No varied matrix parameters available for sampling matrix.")

        n = len(specs)

        values_by_name: dict[str, np.ndarray] = {}
        ticks_by_name: dict[str, dict[int, str] | None] = {}

        for spec in specs:
            values, ticks = _parameter_values(rows, spec)
            values_by_name[spec.name] = values
            ticks_by_name[spec.name] = ticks

        time_scale = _lyapunov_time_axis(use_lyapunov, lambda_max)
        metric_values = _objective_values(rows, metric) * time_scale
        colorbar_label = (
            rf"{_ps().term('validation_horizon')}, $T_h$ [$1/\lambda_{{\max}}$]"
            if use_lyapunov
            else rf"{_ps().term('validation_horizon')}, $T_h$"
        )

        best = _best_row(rows, metric, objective)
        best_values = {spec.name: _parameter_values([best], spec)[0][0] for spec in specs}

        fig, axes = plt.subplots(
            n,
            n,
            figsize=_matrix_figsize(n),
            squeeze=False,
            constrained_layout=False,
        )
        # Slightly wider left margin so categorical y-tick labels (e.g.
        # "Linear + quadratic") sit fully in the margin instead of bleeding into the
        # first column; colorbar pulled closer to the populated matrix.
        fig.subplots_adjust(
            left=_MATRIX_LEFT, right=0.90, bottom=_MATRIX_BOTTOM, top=_MATRIX_TOP, wspace=0.16, hspace=0.16
        )

        scatter_for_colorbar = None

        for i, yspec in enumerate(specs):
            for j, xspec in enumerate(specs):
                ax = axes[i][j]
                x = values_by_name[xspec.name]
                y = values_by_name[yspec.name]

                if i == j:
                    finite_x = x[np.isfinite(x)]
                    bins = min(16, max(5, len(set(finite_x.tolist()))))
                    ax.hist(finite_x, bins=bins)
                    ax.axvline(best_values[xspec.name], color="red", linestyle="--", linewidth=1.5)

                elif i > j:
                    scatter_for_colorbar = ax.scatter(x, y, c=metric_values, s=18, alpha=0.82)
                    ax.scatter(
                        [best_values[xspec.name]],
                        [best_values[yspec.name]],
                        marker="*",
                        s=220,
                        facecolors="red",
                        edgecolors="black",
                        linewidths=1.2,
                        zorder=5,
                    )
                    # Small data margin so a selected star sitting at a parameter
                    # boundary is fully visible instead of clipped by the spine.
                    ax.margins(x=0.08, y=0.08)

                else:
                    ax.axis("off")
                    continue

                ax.grid(False)

                x_ticks = ticks_by_name.get(xspec.name)

                if i == n - 1:
                    # Two-line categorical ticks get a larger label gap so the second
                    # line never runs into the axis label.
                    ax.set_xlabel(
                        _display_param_label(xspec, axis="x"),
                        labelpad=(
                            _XLABEL_PAD_MULTILINE
                            if _is_multiline_category(x_ticks, axis="x")
                            else _XLABEL_PAD_SINGLE_LINE
                        ),
                    )
                else:
                    ax.set_xticklabels([])

                if j == 0 and i != j:
                    ax.set_ylabel(_display_param_label(yspec, axis="y"))
                elif j == 0 and i == j:
                    ax.set_ylabel(_ps().axis_label("Count"))
                else:
                    ax.set_yticklabels([])

                if x_ticks is not None and i == n - 1:
                    # Categorical/boolean x-ticks only on the bottom row (where the
                    # axis label is shown). Upright and centred on the true
                    # categorical positions: long feature-set names are wrapped onto
                    # two lines rather than rotated or shrunk, so every category
                    # reads horizontally at full font size.
                    ax.set_xticks(list(x_ticks.keys()))
                    _labels = ax.set_xticklabels(
                        [_display_category(v, axis="x") for v in x_ticks.values()],
                        rotation=0, ha="center", va="top",
                    )
                    # A tick label is centred on its tick, so the OUTERMOST label
                    # of a panel overhangs by half its own width. At printed cell
                    # width that overhang reaches the neighbouring panel:
                    # "Linear + quadratic" collided with "diagonal" in the next
                    # column. Anchoring the first label at its left edge and the
                    # last at its right edge keeps every label inside its own
                    # panel, whatever the category names are. Spacing cannot fix
                    # this: absorbing the overhang would need wspace = 1.64,
                    # i.e. gaps wider than the panels themselves.
                    if len(_labels) > 1:
                        _labels[0].set_ha("left")
                        _labels[-1].set_ha("right")
                    ax.tick_params(axis="x", pad=_CATEGORICAL_TICK_PAD)
                elif x_ticks is not None:
                    ax.set_xticks(list(x_ticks.keys()))
                    ax.set_xticklabels([])

                y_ticks = ticks_by_name.get(yspec.name)
                if y_ticks is not None and i != j and j == 0:
                    # Y-axis uses the same wrapped form as the x-axis, horizontal and
                    # right-aligned against the axis. multialignment centres the two
                    # lines with respect to each other while the label as a whole
                    # stays right-aligned to the tick.
                    ax.set_yticks(list(y_ticks.keys()))
                    ax.set_yticklabels(
                        [_display_category(v, axis="y") for v in y_ticks.values()],
                        rotation=0, ha="right", va="center", multialignment="center",
                    )
                    ax.tick_params(axis="y", pad=_CATEGORICAL_TICK_PAD)
                elif y_ticks is not None and i != j:
                    ax.set_yticks(list(y_ticks.keys()))
                    ax.set_yticklabels([])

        # Put every bottom-row axis label on a common baseline: the two-line
        # categorical ticks are taller than the numeric ones, which would otherwise
        # leave "Feature set" hanging lower than its neighbours.
        fig.align_xlabels([axes[n - 1][j] for j in range(n)])

        if scatter_for_colorbar is not None:
            # Anchored to the same vertical band as the panel matrix, so the colorbar
            # stays aligned with the panels whatever the bottom margin is.
            cax = fig.add_axes([0.915, _MATRIX_BOTTOM, 0.016, _MATRIX_TOP - _MATRIX_BOTTOM])
            fig.colorbar(scatter_for_colorbar, cax=cax, label=colorbar_label)

        # The 0.3 in outer margin was chosen when these canvases were 12-17 in
        # wide, where it was a small relative border. At the printed size it is
        # a tenth of the width, and because it is part of the saved image it is
        # scaled down with the figure on inclusion, shrinking the type. The
        # contract's own default is used instead.
        _fold_offsets_for_figure(fig)

        return save_figure(fig, output_base, formats, dpi,
                           pad_inches=_ps().style()["output"]["rc_common"]
                           ["savefig.pad_inches"])


def _surrogate_specs_with_limit(rows: list[dict[str, Any]], specs: list[ParameterSpec], *, max_specs: int = 4) -> list[ParameterSpec]:
    specs = varied_specs_for_surrogate(rows, specs)

    if len(specs) <= max_specs:
        return specs

    priority = [
        "context_steps",
        "delay_spacing",
        "ridge_alpha",
        "feature_dimension",
        "reservoir_size",
        "spectral_radius",
        "input_scaling",
        "leaking_rate",
        "regularization",
        "learning_rate",
    ]

    chosen: list[ParameterSpec] = []

    for name in priority:
        for spec in specs:
            if spec.name == name and spec not in chosen:
                chosen.append(spec)
                break
        if len(chosen) == max_specs:
            return chosen

    for spec in specs:
        if spec not in chosen:
            chosen.append(spec)
        if len(chosen) == max_specs:
            return chosen

    return chosen


def plot_surrogate_landscape(
    rows: list[dict[str, Any]],
    *,
    specs: list[ParameterSpec],
    metric: str,
    objective: str,
    output_base: Path,
    formats: list[str],
    dpi: int,
    max_train_points: int = 1200,
    random_seed: int = 0,
    x_parameter: str | None = None,
    y_parameter: str | None = None,
    use_lyapunov: bool = False,
    lambda_max: float = 1.0,
) -> list[str]:
    try:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise RuntimeError(
            "Surrogate landscape requires scikit-learn. Install scikit-learn or disable this plot."
        ) from exc

    # Type sizes come from the contract, not from this module.
    with plt.rc_context(_ps().rc_params(_FAMILY_SURROGATE)):
        rows = ok_rows(rows, metric)
        specs = _surrogate_specs_with_limit(rows, specs, max_specs=4)

        if len(specs) < 1:
            raise ValueError("No varied numerical parameters available for surrogate plot.")

        rng = np.random.default_rng(random_seed)

        if len(rows) > max_train_points:
            indices = rng.choice(len(rows), size=max_train_points, replace=False)
            train_rows = [rows[int(i)] for i in sorted(indices)]
        else:
            train_rows = rows

        x_columns: list[np.ndarray] = []

        for spec in specs:
            values, _ = _parameter_values(train_rows, spec)
            x_columns.append(values)

        x = np.column_stack(x_columns)
        y = _objective_values(train_rows, metric)

        finite = np.isfinite(x).all(axis=1) & np.isfinite(y)
        x = x[finite]
        y = y[finite]

        if len(y) < max(8, len(specs) + 3):
            raise ValueError("Too few finite optimizer rows for surrogate plot.")

        time_scale = _lyapunov_time_axis(use_lyapunov, lambda_max)
        y_scaled = y * time_scale

        # Wrapped onto two lines. This label sits on the top-left diagonal panel
        # and, as one line, was taller than that panel: its first character ran
        # into the y tick labels of the panel below. Same remedy as the wrapped
        # parameter names -- a rotated label grows ALONG its panel, so wrapping
        # shortens it where the room is short.
        diag_label = (
            rf"{_ps().term('validation_horizon')}, $T_h$" "\n" rf"[$1/\lambda_{{\max}}$]"
            if use_lyapunov
            else rf"{_ps().term('validation_horizon')}," "\n" rf"$T_h$"
        )
        colorbar_label = (
            rf"{_ps().term('surrogate_validation_horizon')}, $T_h$ [$1/\lambda_{{\max}}$]"
            if use_lyapunov
            else rf"{_ps().term('surrogate_validation_horizon')}, $T_h$"
        )

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=np.ones(x.shape[1]))
            + WhiteKernel(noise_level=1e-5)
        )

        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=random_seed,
            n_restarts_optimizer=2,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x_scaled, y_scaled)

        best = _best_row(train_rows, metric, objective)
        best_x = np.asarray([_parameter_values([best], spec)[0][0] for spec in specs], dtype=float)
        center = best_x.copy()

        n = len(specs)
        fig, axes = plt.subplots(
            n,
            n,
            figsize=_surrogate_figsize(n),
            squeeze=False,
            constrained_layout=False,
        )
        # The colourbar is placed absolutely, so the space reserved to its right
        # is a FRACTION of the canvas. At the old 12 in width the leftover 0.069
        # was 0.83 in and the rotated colourbar label fitted; at the printed
        # 5.9 in it is only 0.41 in, and the label plus its ticks overflowed the
        # figure by 0.17 in. bbox_inches="tight" then grew the saved canvas, and
        # including that at the class width would have shrunk every glyph by 9 %.
        # wspace 0.24, not 0.16: at printed size the outermost numeric tick label
        # of each panel overhung far enough to touch its neighbour's. Measured
        # sweep: 0.16 -> 3 collisions, 0.24 -> 0.
        # left 0.145: at 5.9 in the old 0.11 gave 0.65 in, which a two-line
        # rotated y-label plus its tick labels overran.
        fig.subplots_adjust(left=0.145, right=0.855, bottom=0.10, top=0.92, wspace=0.24, hspace=0.18)

        image_for_colorbar = None

        for i, yspec in enumerate(specs):
            for j, xspec in enumerate(specs):
                ax = axes[i][j]

                if i < j:
                    ax.axis("off")
                    continue

                if i == j:
                    x_values = x[:, j]
                    grid = np.linspace(np.nanmin(x_values), np.nanmax(x_values), 160)
                    x_grid = np.tile(center, (len(grid), 1))
                    x_grid[:, j] = grid

                    pred = model.predict(scaler.transform(x_grid))

                    ax.plot(grid, pred, linewidth=1.7)
                    ax.scatter(x[:, j], y_scaled, s=12, alpha=0.38)
                    ax.axvline(best_x[j], color="red", linestyle="--", linewidth=1.4)
                    ax.set_ylabel(diag_label if j == 0 else "")

                else:
                    x_values = x[:, j]
                    y_values = x[:, i]

                    x_grid_1d = np.linspace(np.nanmin(x_values), np.nanmax(x_values), 70)
                    y_grid_1d = np.linspace(np.nanmin(y_values), np.nanmax(y_values), 70)
                    xx, yy = np.meshgrid(x_grid_1d, y_grid_1d)

                    x_grid = np.tile(center, (xx.size, 1))
                    x_grid[:, j] = xx.ravel()
                    x_grid[:, i] = yy.ravel()

                    pred = model.predict(scaler.transform(x_grid)).reshape(xx.shape)

                    image_for_colorbar = ax.imshow(
                        pred,
                        origin="lower",
                        aspect="auto",
                        extent=[x_grid_1d.min(), x_grid_1d.max(), y_grid_1d.min(), y_grid_1d.max()],
                        # Rasterised at figure dpi so the PDF carries the same image the
                        # PNG shows. Without this the raw array is embedded and the viewer
                        # smooths it up to page size, softening every cell and washing out
                        # colour differently in each viewer.
                        rasterized=True,
                    )
                    ax.scatter(x[:, j], x[:, i], s=9, alpha=0.35)
                    ax.scatter(
                        [best_x[j]],
                        [best_x[i]],
                        marker="*",
                        s=155,
                        facecolors="red",
                        edgecolors="black",
                        linewidths=1.2,
                        zorder=5,
                    )
                    # Expand the view slightly beyond the imshow extent so a star at
                    # a parameter boundary is fully visible instead of clipped.
                    xr = float(x_grid_1d.max() - x_grid_1d.min()) or 1.0
                    yr = float(y_grid_1d.max() - y_grid_1d.min()) or 1.0
                    ax.set_xlim(x_grid_1d.min() - 0.04 * xr, x_grid_1d.max() + 0.04 * xr)
                    ax.set_ylim(y_grid_1d.min() - 0.04 * yr, y_grid_1d.max() + 0.04 * yr)

                ax.grid(False)

                if i == n - 1:
                    ax.set_xlabel(_display_param_label(xspec, axis="x"))
                else:
                    ax.set_xticklabels([])

                if j == 0 and i != j:
                    ax.set_ylabel(_display_param_label(yspec, axis="y"))
                elif j != 0:
                    ax.set_yticklabels([])

        if image_for_colorbar is not None:
            cax = fig.add_axes([0.875, 0.10, 0.016, 0.82])
            colorbar = fig.colorbar(cax=cax, mappable=image_for_colorbar, label=colorbar_label)
            # Display-only: the colorbar axes is placed absolutely, so relabelling
            # the ticks cannot move it or the panel matrix.
            _apply_colorbar_tick_format(colorbar)

        # The 0.3 in outer margin was chosen when these canvases were 12-17 in
        # wide, where it was a small relative border. At the printed size it is
        # a tenth of the width, and because it is part of the saved image it is
        # scaled down with the figure on inclusion, shrinking the type. The
        # contract's own default is used instead.
        _fold_offsets_for_figure(fig)

        return save_figure(fig, output_base, formats, dpi,
                           pad_inches=_ps().style()["output"]["rc_common"]
                           ["savefig.pad_inches"])
