#!/usr/bin/env python3
"""Per-model seed-robustness candidate boxplots (public, reusable).

Renders one boxplot figure per seed-reranked stochastic model directly from the
staged provenance CSVs. Reads CSV only: no model is loaded, no metric is
recomputed, no candidate is reselected. The winner is *read* from the selection
record, never re-derived.

This is a public port of the historical private ``plot_seed_robust_per_model.py``
with the approved publication conventions applied: no descriptive overall title,
legend entries "Winner" / "Published candidate" / "Other candidates", and a
right-edge annotation kept inside the axes with a safe margin. Input and output
directories are explicit CLI arguments -- no private path is assumed.

The horizon axis stays in physical simulation time (``T_h``); it is not a
Lyapunov-time axis, so ``--time-axis-mode`` does not apply here.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

_SCRIPT = Path(__file__).resolve()
_REPO_ROOT = _SCRIPT.parents[2]
_FORECASTING_SRC = _REPO_ROOT / "forecasting" / "src"
for _bootstrap in (_REPO_ROOT, _FORECASTING_SRC):
    if str(_bootstrap) not in sys.path:
        sys.path.insert(0, str(_bootstrap))

from common import plot_style as PS  # noqa: E402
from common.logging_setup import get_logger  # noqa: E402
from common.plotting import model_display_name  # noqa: E402
from forecasting.visualization.style import configure_matplotlib, save_figure  # noqa: E402

logger = get_logger("forecasting.scripts.plot_seed_robustness")

# The six stochastic models that went through a 15-seed rerank. NGRC is deterministic
# (no RNG -> no seed figure). pinn_full64 joined this set once its v003 physics-active
# study replaced the earlier staged rerank with a uniform 10 x 15 grid; it has no
# published-candidate row, so its figure omits the published marker (see plot_model).
SEED_MODELS: tuple[str, ...] = (
    "rc_full64",
    "rc_latent8",
    "perc_full64",
    "perc_latent8",
    "pinn_latent8",
    "pinn_full64",
)

N_CANDIDATES = 10
HORIZON_COLUMN = "valid_relative_l2_horizon_time"
HORIZON_LABEL = rf"{PS.term('validation_horizon')}, $T_h$"

# Colourblind-aware (Okabe-Ito): winner green, published vermillion, other blue.
# Okabe-Ito throughout: the third colour was previously #9ecae1, which is not
# an Okabe-Ito colour despite the palette being documented as such.
COLOR_WINNER = PS.colour("bluish_green")
COLOR_PUBLISHED = PS.colour("vermillion")
COLOR_OTHER = PS.colour("sky_blue")

LEGEND_ENTRIES = ("Winner", "Published candidate", "Other candidates")

_DIST_CSV = "seed_robust_candidate_distributions.csv"
_SELECTED_CSV = "all_models_selected_candidates.csv"
_PUBLISHED_CSV = "old_vs_seedrobust_validation_comparison.csv"


def nice_model_name(model_id: str) -> str:
    """'rc_full64' -> 'RC-64', 'pinn_latent8' -> 'PINN-8' (display metadata only)."""
    stem, _, representation = model_id.rpartition("_")
    return model_display_name(stem.upper(), representation)


def _read_rows(input_dir: Path, name: str) -> list[dict[str, str]]:
    path = input_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing seed-robustness provenance table: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_candidate_distributions(rows: list[dict[str, str]]) -> dict[str, dict[int, list[float]]]:
    """model_id -> {candidate_rank -> [15 seed horizons]} in stable rank order."""
    by_model: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_model[row["model_id"]][int(row["candidate_rank"])].append(float(row[HORIZON_COLUMN]))
    return by_model


def ordered_candidates(candidates: dict[int, list[float]]) -> list[list[float]]:
    """Deterministic candidate order: search rank 1..N_CANDIDATES."""
    return [candidates[rank] for rank in range(1, N_CANDIDATES + 1)]


def plot_model(
    model_id: str,
    candidates: dict[int, list[float]],
    winner_rank: int,
    published_value: float | None,
    output_dir: Path,
    *,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> list[str]:
    """Render one seed-robustness boxplot. No overall title; approved legend.

    ``published_value`` is None for models whose previously published configuration is not
    one of the candidates shown. Marking rank 1 as the published candidate would be false
    for those models, so both the reference line and the published styling are omitted.
    """
    formats = formats or ["png"]
    data = ordered_candidates(candidates)
    show_published = published_value is not None

    # Type sizes from the contract, not inherited from the global state.
    with plt.rc_context(PS.rc_params("seed_robust_boxplots")):
        # constrained_layout so the axes fill the declared canvas. Without it the
        # default margins leave enough slack that bbox_inches="tight" trimmed the
        # saved figure to 5.2 in, and including that at the class width scaled
        # every glyph UP by 13 % -- the declared type size again not being the
        # size delivered.
        fig, ax = plt.subplots(figsize=PS.figure_size_in("seed_robust_boxplots"),
                               constrained_layout=True)
        boxes = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.62,
            medianprops=dict(color="black", lw=1.4),
            flierprops=dict(marker="o", ms=3.0, alpha=0.5),
        )
        for i, box in enumerate(boxes["boxes"], start=1):
            if i == winner_rank:
                box.set_facecolor(COLOR_WINNER)
            elif i == 1 and show_published:
                box.set_facecolor(COLOR_PUBLISHED)
            else:
                box.set_facecolor(COLOR_OTHER)
            box.set_alpha(0.9)

        if show_published:
            ax.axhline(published_value, color=COLOR_PUBLISHED, ls="--", lw=1.3)
            # Annotation kept inside the axes with a safe margin (axes-fraction x=0.985),
            # so it never sits on the extreme right edge.
            ax.annotate(
                f"published {published_value:.1f}",
                xy=(0.985, published_value),
                xycoords=("axes fraction", "data"),
                fontsize=PS.rc_params("seed_robust_boxplots")["xtick.labelsize"],
                color=COLOR_PUBLISHED,
                ha="right",
                va="bottom",
            )

        # No descriptive overall title (approved). Model identity lives in the output
        # path and the returned metadata, not in a plot title.
        ax.set_xlabel("candidate (search rank)")
        ax.set_ylabel(HORIZON_LABEL)
        ax.set_xticks(range(1, N_CANDIDATES + 1))
        ax.set_xticklabels([str(c) for c in range(1, N_CANDIDATES + 1)])
        ax.grid(axis="y", alpha=0.25)

        colors = [COLOR_WINNER, COLOR_PUBLISHED, COLOR_OTHER]
        labels = list(LEGEND_ENTRIES)
        if not show_published:
            # drop the "Published candidate" entry rather than show a swatch for absent styling
            colors = [COLOR_WINNER, COLOR_OTHER]
            labels = [LEGEND_ENTRIES[0], LEGEND_ENTRIES[2]]
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
        ax.legend(
            handles,
            labels,
            frameon=False,
            fontsize=PS.rc_params("seed_robust_boxplots")["legend.fontsize"],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
            ncol=len(labels),
        )

        return save_figure(fig, output_dir / model_id / "seed_robust_candidate_boxplots", formats, dpi)


def generate_all(input_dir: Path, output_dir: Path, *, dpi: int = 300) -> list[dict[str, Any]]:
    """Render every supported seed-robustness model; return per-model metadata."""
    configure_matplotlib()

    by_model = load_candidate_distributions(_read_rows(input_dir, _DIST_CSV))
    selected = {r["model_id"]: r for r in _read_rows(input_dir, _SELECTED_CSV)}
    published = {r["model_id"]: r for r in _read_rows(input_dir, _PUBLISHED_CSV)}

    results: list[dict[str, Any]] = []
    for model_id in SEED_MODELS:
        candidates = by_model.get(model_id)
        if not candidates:
            raise ValueError(f"No rerank rows found for {model_id} in {input_dir}.")
        winner_rank = int(selected[model_id]["selected_candidate_rank"])
        # Absent by design for models whose published configuration predates this candidate
        # set; plot_model then omits the published line and styling.
        published_row = published.get(model_id)
        published_value = (
            float(published_row["old_selected_valid_h"]) if published_row else None
        )
        paths = plot_model(model_id, candidates, winner_rank, published_value, output_dir, dpi=dpi)
        results.append(
            {
                "model_id": model_id,
                "display_name": nice_model_name(model_id),
                "winner_rank": winner_rank,
                "n_candidates": len(candidates),
                "has_published_marker": published_value is not None,
                "paths": paths,
            }
        )
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render per-model seed-robustness candidate boxplots from staged "
            "provenance CSVs (public, deterministic; reads CSV only)."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help=(
            "Directory containing the staged provenance CSVs "
            f"({_DIST_CSV}, {_SELECTED_CSV}, {_PUBLISHED_CSV})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write <model>/seed_robust_candidate_boxplots.png under.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Saved raster resolution (default 300).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = generate_all(args.input_dir, args.output_dir, dpi=args.dpi)
    for meta in results:
        logger.info(
            "Seed figure | model=%s display=%s winner=c%02d candidates=%d output=%s",
            meta["model_id"], meta["display_name"], meta["winner_rank"],
            meta["n_candidates"], meta["paths"][0],
        )
    logger.info(
        "Seed-robustness done | figures=%d (pinn_full64 and ngrc excluded by design)",
        len(results),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
