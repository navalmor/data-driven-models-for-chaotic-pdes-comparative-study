#!/usr/bin/env python3
"""Generator for the PINN-64 paired physics-weight ablation figure.

WHY THIS FILE EXISTS
    The published figure
    ``plots/comparison/pinn_full64_physics_ablation/paired_active_vs_zero_weight.png``
    had no generator in the repository. It was produced by a one-off packaging
    step (``package_generated_from_staged_provenance_csv``, stage 7b) whose code
    was never committed, so the only figure in the package that could not be
    re-executed was also the one carrying an unrounded 17-digit weight in its
    tick labels. This script closes that gap.

WHAT IT DOES AND DOES NOT DO
    Every number is read from the tracked provenance files. Nothing is
    recomputed, no model is run, no metric is recalculated:

        paired_seed_results.csv   one row per seed, both arms, and their difference
        paired_summary.json       the medians and win counts, used only to verify

    The script asserts its own plotted values against ``paired_summary.json``
    before saving, so a silent divergence between figure and record is not
    possible.

SCOPE OF THE EVIDENCE (repeated here because the figure is easy to over-read)
    Validation split only, by design; no active-versus-zero comparison exists on
    the held-out test split. One configuration, whose hyperparameters were
    themselves tuned with the physics term active, which favours the active arm.
    Descriptive, not a significance test.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
import sys
from pathlib import Path

import matplotlib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import plot_style as PS  # noqa: E402
from common.logging_setup import get_logger  # noqa: E402

logger = get_logger("forecasting.scripts.plot_physics_ablation")

FAMILY = "physics_ablation"
DEFAULT_INPUT = ("final_thesis_package_v001/results/02_forecasting/provenance/"
                 "pinn_full64_physics_ablation")
DEFAULT_OUTPUT = ("final_thesis_package_v001/plots/comparison/"
                  "pinn_full64_physics_ablation/paired_active_vs_zero_weight")


def load(input_dir: Path) -> tuple[list[dict], dict]:
    rows = list(csv.DictReader((input_dir / "paired_seed_results.csv").open()))
    summary = json.loads((input_dir / "paired_summary.json").read_text())
    if not rows:
        raise ValueError(f"no paired rows in {input_dir}")
    return rows, summary


def _physics_weight(summary: dict) -> float:
    """The active arm's physics weight, parsed from the recorded design string."""
    design = str(summary.get("design", ""))
    for token in design.replace("(", " ").replace(")", " ").split():
        try:
            value = float(token)
        except ValueError:
            continue
        if 0.0 < value < 1.0:
            return value
    raise ValueError("could not read lambda_phy from paired_summary.json design")


def _sci(value: float, sig: int = 3) -> str:
    """LaTeX scientific notation to a stated number of significant figures."""
    mantissa, exponent = f"{value:.{sig - 1}e}".split("e")
    return rf"{mantissa} \times 10^{{{int(exponent)}}}"


def build(rows: list[dict], summary: dict):
    seeds = [r["seed"] for r in rows]
    active = np.array([float(r["active_valid_horizon_time"]) for r in rows])
    zero = np.array([float(r["zero_weight_valid_horizon_time"]) for r in rows])
    diff = np.array([float(r["paired_difference_active_minus_zero"]) for r in rows])

    med_a, med_z, med_d = stats.median(active), stats.median(zero), stats.median(diff)
    n_up, n_dn = int((diff > 0).sum()), int((diff < 0).sum())
    lam = _physics_weight(summary)

    # The figure must agree with the record it was drawn from.
    assert med_a == summary["active_vph_median"], "median active arm disagrees"
    assert n_up == summary["active_better_seeds"], "win count disagrees"
    assert round(med_d, 1) == round(
        summary["median_paired_difference_active_minus_zero"], 1), "median difference"

    c_active = PS.colour("bluish_green")
    c_zero = PS.colour("vermillion")

    # pyplot is imported here rather than at module scope so the Agg backend is
    # selected first, but it must be bound BEFORE rc_context is called --
    # matplotlib.pyplot is not an attribute of matplotlib until it is imported,
    # so build() raised AttributeError whenever it was called outside main().
    import matplotlib.pyplot as plt

    with plt.rc_context(PS.rc_params(FAMILY)):
        fig, axes = plt.subplots(1, 2, figsize=PS.figure_size_in(FAMILY),
                                 constrained_layout=True)
        # The two panels carry a rotated y-label each, so the default constrained
        # spacing leaves them visually crowded against one another.
        fig.get_layout_engine().set(w_pad=0.06, wspace=0.06)
        fig.suptitle("PINN-64 — validation only", fontweight="bold")

        # ---- (a) paired slopes ------------------------------------------
        ax = axes[0]
        for a, z, d in zip(active, zero, diff):
            ax.plot([0, 1], [a, z], color=c_active if d > 0 else c_zero,
                    linewidth=1.1, marker="o", markersize=3.2, zorder=2)
        # Anchored in AXES fractions, not data coordinates. Deriving the x limits
        # from the drawn text is a fixed point that does not converge: widening
        # the limits moves the anchor inward while the text keeps its width, so
        # the label can still cross the frame. Anchoring at 2 % and 98 % of the
        # axes with the matching alignment keeps both labels inside by
        # construction, whatever the values or the type size.
        ax.hlines(med_a, -0.16, 0.16, color="black", linewidth=2.4, zorder=3)
        ann_a = ax.annotate(f"median {med_a:.1f}", xy=(0.02, med_a),
                            xycoords=("axes fraction", "data"),
                            ha="left", va="bottom")
        ax.hlines(med_z, 0.84, 1.16, color="black", linewidth=2.4, zorder=3)
        ann_z = ax.annotate(f"median {med_z:.1f}", xy=(0.98, med_z),
                            xycoords=("axes fraction", "data"),
                            ha="right", va="bottom")
        ax.set_xticks([0, 1], [
            "active" "\n" rf"($\lambda_{{\mathrm{{phy}}}} = {_sci(lam)}$)",
            "zero-weight" "\n" r"($\lambda_{\mathrm{phy}} = 0$)",
        ])
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylabel(rf"{PS.term('validation_horizon')}, $T_h$")
        # The letters stay inside their descriptive titles: this figure has never
        # used standalone boxed panel labels and the panel-label policy is to
        # preserve what each family already does.
        ax.set_title(f"(a) paired by seed, {len(rows)} seeds")
        ax.grid(True, axis="y", alpha=0.25)

        # ---- (b) paired differences --------------------------------------
        ax = axes[1]
        x = np.arange(len(rows))
        ax.bar(x, diff, color=[c_active if d > 0 else c_zero for d in diff], zorder=2)
        ax.axhline(0, color="black", linewidth=0.9, zorder=3)
        ax.axhline(med_d, color="black", linestyle="--", linewidth=1.2, zorder=3)
        ax.annotate(f"median {med_d:+.1f}", xy=(len(rows) - 1.15, med_d),
                    ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor="none", alpha=0.75))
        # Fifteen two-digit labels across a panel this narrow leave no gap
        # between them and read as one run-on string, "0102030405...". Rotated
        # upright they are separated by the full tick spacing instead.
        ax.set_xticks(x, [s[-2:] for s in seeds])
        ax.tick_params(axis="x", rotation=90)
        ax.set_xlabel(PS.axis_label(f"seed ({seeds[0][:5]}xx)"))
        # Wrapped: as one line this rotated label is taller than its own panel
        # and crowds the title above it.
        ax.set_ylabel(PS.axis_label("paired difference,\nactive − zero-weight"))
        # Wrapped, not shortened: at the printed panel width this title is 39 %
        # wider than the axes it names and overflowed the figure by 0.40 in,
        # which bbox_inches="tight" then absorbed by growing the saved canvas.
        # Both counts are kept because 11/15 and 4/15 are not complementary --
        # ties are reported separately in the summary.
        ax.set_title(f"(b) active better on {n_up}/{len(rows)},\n"
                     f"zero-weight on {n_dn}/{len(rows)}")
        ax.grid(True, axis="y", alpha=0.25)

    return fig, {"median_active": med_a, "median_zero": med_z,
                 "median_difference": round(med_d, 4),
                 "active_better": n_up, "zero_better": n_dn,
                 "n_pairs": len(rows), "lambda_phy": lam}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default=DEFAULT_INPUT)
    ap.add_argument("--output-base", default=DEFAULT_OUTPUT,
                    help="path without extension; one file per requested format")
    ap.add_argument("--formats", nargs="+", default=["png"])
    ap.add_argument("--dpi", type=int, default=None)
    args = ap.parse_args()

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = REPO_ROOT / input_dir
    out_base = Path(args.output_base)
    if not out_base.is_absolute():
        out_base = REPO_ROOT / out_base
    out_base.parent.mkdir(parents=True, exist_ok=True)
    dpi = args.dpi or PS.style()["output"]["dpi"]

    rows, summary = load(input_dir)
    written = []
    for fmt in args.formats:
        # One separate render instance per format, from the same data and the
        # same specification -- never one figure saved twice.
        fig, values = build(rows, summary)
        dest = out_base.with_suffix(f".{fmt}")
        fig.savefig(dest, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        written.append(str(dest))

    logger.info("physics ablation figure written | files=%s | values=%s",
                written, json.dumps(values, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
