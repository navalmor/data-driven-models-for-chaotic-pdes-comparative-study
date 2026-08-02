"""Central plotting size and style contract (Phase 2, CH-12 / CH-13).

Every size, type size, colour, marker style and notation choice used by a
published or thesis figure is declared once, in ``plot_style.json``, and read
through this module. Nothing here computes or alters a scientific value; it
decides only how a value is drawn.

WHY FIGURE SIZE IS PART OF THE STYLE CONTRACT
    A declared type size is only meaningful if the canvas is the printed canvas.
    The optimizer sampling matrix was drawn 16.79 in wide and included at
    7.60 in, so LaTeX rescaled it by 0.453 and its 10 pt tick labels reached
    paper at 4.53 pt. Every published family had the same problem to a lesser
    degree. :func:`figure_size_in` therefore returns the printed size, and
    :func:`downscale_factor` exists so a test can prove no figure is rescaled.

NOTATION
    ``T_h`` is a physical simulation time and must never appear on a
    Lyapunov-scaled axis; ``t_lambda`` is a position on a Lyapunov-scaled axis
    and must never appear on a physical one. :func:`horizon_symbol` and
    :func:`cutoff_symbol` are the only sanctioned way to pick between them.

PANEL LABELS
    The existing policy is preserved, not unified: families that already carry
    boxed a/b/c labels keep them, families that carry none stay without, and no
    label is added for uniformity. :func:`panel_label_policy` reports which is
    which so a test can assert the sets never change.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = [
    "STYLE_PATH", "style", "page_class", "family", "families",
    "figure_size_in", "rc_params", "marker", "palette", "colour",
    "horizon_symbol", "cutoff_symbol", "horizon_label", "cutoff_label",
    "term", "retired_terms", "axis_label", "pdf_class", "panel_label_policy",
    "downscale_factor", "minimum_printed_pt", "effective_printed_pt",
    "scatter_size", "star_size",
]

STYLE_PATH = Path(__file__).with_name("plot_style.json")


@lru_cache(maxsize=1)
def style() -> dict[str, Any]:
    """The parsed style contract. Cached; the file is read once per process."""
    return json.loads(STYLE_PATH.read_text())


# --------------------------------------------------------------------------
# lookup
# --------------------------------------------------------------------------
def families() -> list[str]:
    return [k for k in style()["families"] if not k.startswith("_")]


def family(name: str) -> dict[str, Any]:
    fams = style()["families"]
    if name not in fams or name.startswith("_"):
        raise KeyError(f"unknown figure family {name!r}; known: {families()}")
    return fams[name]


def page_class(name: str) -> dict[str, Any]:
    classes = style()["page_classes"]
    if name not in classes:
        raise KeyError(f"unknown page class {name!r}; known: {sorted(classes)}")
    return classes[name]


def palette() -> dict[str, str]:
    return {k: v for k, v in style()["palette"].items() if not k.startswith("_")}


def colour(name: str) -> str:
    try:
        return palette()[name]
    except KeyError:
        raise KeyError(f"{name!r} is not in the Okabe-Ito palette: {sorted(palette())}")


def marker(kind: str) -> dict[str, Any]:
    marks = style()["markers"]
    if kind not in marks or kind.startswith("_"):
        raise KeyError(f"unknown marker kind {kind!r}; "
                       f"known: {[k for k in marks if not k.startswith('_')]}")
    return dict(marks[kind])


def pdf_class(family_name: str) -> str:
    """``'vector'`` or ``'hybrid'`` -- whether the PDF embeds a raster layer."""
    return family(family_name)["pdf_class"]


def panel_label_policy() -> dict[str, list[str]]:
    p = style()["panel_labels"]
    return {k: list(v) for k, v in p.items() if isinstance(v, list)}


def minimum_printed_pt() -> dict[str, float]:
    return {k: v for k, v in style()["minimum_printed_pt"].items()
            if not k.startswith("_")}


# --------------------------------------------------------------------------
# size
# --------------------------------------------------------------------------
def figure_size_in(family_name: str, *, aspect_ratio: float | None = None
                   ) -> tuple[float, float]:
    """Printed ``(width, height)`` in inches for one figure family.

    ``aspect_ratio`` (height / width) is required only for families whose shape
    depends on the data -- the two optimizer matrices, whose panel grid grows
    with the number of varied search parameters.
    """
    fam = family(family_name)
    cls = page_class(fam["page_class"])
    width = float(cls["width_in"])

    declared = fam["aspect_ratio"]
    if declared == "derived":
        if aspect_ratio is None:
            raise ValueError(
                f"{family_name} has a data-dependent shape; pass aspect_ratio")
        ratio = float(aspect_ratio)
    else:
        ratio = float(declared) if aspect_ratio is None else float(aspect_ratio)

    height = width * ratio
    cap = cls.get("max_height_in")
    if cap is not None:
        height = min(height, float(cap))
    return width, height


def downscale_factor(family_name: str, drawn_width_in: float) -> float:
    """Printed width / drawn width.

    1.0 means the figure is included at the size it was drawn, which is the
    only value at which a declared type size survives to paper. Anything below
    1.0 shrinks every glyph in the figure by that factor.
    """
    if drawn_width_in <= 0:
        raise ValueError(f"drawn_width_in must be positive, got {drawn_width_in}")
    printed, _ = figure_size_in(
        family_name,
        aspect_ratio=1.0 if family(family_name)["aspect_ratio"] == "derived" else None)
    return printed / float(drawn_width_in)


def effective_printed_pt(family_name: str, declared_pt: float,
                         drawn_width_in: float) -> float:
    """The type size a reader actually gets, after any inclusion rescaling."""
    return float(declared_pt) * downscale_factor(family_name, drawn_width_in)


def _type_scale(family_name: str) -> dict[str, Any]:
    return style()["type_scales"][page_class(family(family_name)["page_class"])["type_scale"]]


def scatter_size(family_name: str) -> float:
    dense = page_class(family(family_name)["page_class"])["type_scale"] == "dense"
    m = style()["markers"]["trial_point"]
    return float(m["size_dense"] if dense else m["size_standard"])


def star_size(family_name: str) -> float:
    dense = page_class(family(family_name)["page_class"])["type_scale"] == "dense"
    m = style()["markers"]["best_trial_star"]
    return float(m["size_dense"] if dense else m["size_standard"])


def rc_params(family_name: str, **overrides: Any) -> dict[str, Any]:
    """Matplotlib rcParams for a family: its type scale plus the common output rc.

    Pass the result to ``plt.rc_context`` around the whole of a plotting
    function, so that no call site needs to name a font size.
    """
    rc: dict[str, Any] = dict(_type_scale(family_name))
    rc.update(style()["output"]["rc_common"])
    rc.update(overrides)
    return rc


# --------------------------------------------------------------------------
# notation and terminology
# --------------------------------------------------------------------------
def horizon_symbol(use_lyapunov: bool) -> str:
    n = style()["notation"]
    return n["horizon_symbol_lyapunov"] if use_lyapunov else n["horizon_symbol_physical"]


def cutoff_symbol(use_lyapunov: bool) -> str:
    n = style()["notation"]
    return n["cutoff_symbol_lyapunov"] if use_lyapunov else n["cutoff_symbol_physical"]


def _fmt(value: float) -> str:
    return style()["notation"]["value_format"].format(float(value))


def horizon_label(value: float, use_lyapunov: bool, *, quantity: str = "prediction_horizon"
                  ) -> str:
    """Legend text for a horizon marker, e.g. ``Prediction horizon, $t_\\lambda = 1.05$``.

    ``quantity`` selects the noun phrase. ``prediction_horizon`` is a forecast
    horizon; ``reconstruction_validity_time`` is the autoencoder round-trip
    quantity, which is a different measurement and must not borrow the forecast
    name -- see ``docs/horizon_quantities.md``.
    """
    return f"{term(quantity)}, ${horizon_symbol(use_lyapunov)} = {_fmt(value)}$"


def cutoff_label(value: float, use_lyapunov: bool) -> str:
    return f"{term('safety_cutoff')}, ${cutoff_symbol(use_lyapunov)} = {_fmt(value)}$"


def term(key: str) -> str:
    t = style()["terminology"]
    if key not in t or key in ("retired", "_comment"):
        raise KeyError(f"unknown term {key!r}; "
                       f"known: {[k for k in t if k not in ('retired', '_comment')]}")
    return t[key]


def retired_terms() -> list[str]:
    return list(style()["terminology"]["retired"])


def axis_label(text: str) -> str:
    """Apply the axis-label case rule.

    Lowercase sentence case, but never touching a label that opens with LaTeX
    maths or an already-lowercase word, and never lowercasing an interior
    capital such as ``L^2``.
    """
    if not text or style()["labels"]["axis_label_case"] != "lower":
        return text
    if text[0] in "$\\":
        return text
    return text[0].lower() + text[1:]
