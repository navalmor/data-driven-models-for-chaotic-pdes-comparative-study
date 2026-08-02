from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def configure_matplotlib() -> None:
    """Apply compact, publication-friendly matplotlib defaults."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.constrained_layout.use": True,
        }
    )


def save_figure(
    fig: plt.Figure,
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
    *,
    pad_inches: float = 0.04,
) -> list[str]:
    # Default to a small nonzero outer padding: bbox_inches="tight" with
    # pad_inches=0.0 can shave anti-aliased glyph edges (axis labels, the bottom
    # time label). Callers that need more breathing room (the optimiser matrices)
    # still pass an explicit larger value.
    output_base.parent.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    # Output rcParams must be in force at SAVE time, not at draw time. Font
    # embedding in particular is read by the backend when the file is written,
    # so wrapping only the plotting code leaves pdf.fonttype at its default and
    # every PDF silently comes out with Type 3 fonts -- which drop the Greek on
    # text extraction and are poorly regarded for print.
    from common import plot_style as _ps

    with plt.rc_context(_ps.style()["output"]["rc_common"]):
        for fmt in formats:
            suffix = fmt.lower().lstrip(".")
            path = output_base.with_suffix(f".{suffix}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
            written.append(str(path))

    plt.close(fig)
    return written
