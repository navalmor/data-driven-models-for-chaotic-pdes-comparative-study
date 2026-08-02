# Optimiser search

Each forecasting model has an optimiser-search script that tunes its hyperparameters by
Bayesian optimisation. These scripts are included so the method can be inspected and reused on
new data. The historical searches behind the thesis results ran on much larger budgets and
their raw output is not part of this repository; the scripts here demonstrate the interface,
not those searches.

The scripts are
[`forecasting/scripts/optimizer_search_ngrc.py`](../forecasting/scripts/optimizer_search_ngrc.py),
`optimizer_search_rc.py`, `optimizer_search_perc.py`, and `optimizer_search_pinn.py`. Ready-to-run
examples are in
[`forecasting/examples/optimizer_search/`](../forecasting/examples/optimizer_search/); see that
directory's README for commands and for how to point a search at your own data.

## Backend and objective

The searches use scikit-optimize. Each search maximises a validation objective based on the
model's relative-L2 horizon time — the time until the prediction's relative L2 error crosses a
threshold — evaluated on the validation split. The test split is never used during a search.

Each search runs several optimiser families over the same space: a Gaussian-process surrogate
(`gp`), random sampling (`dummy`), and tree-based surrogates (`forest`, `gbrt`). Random sampling
is included as a baseline, since a surrogate that cannot beat random search is not helping.

## Seeds and outputs

Seeds are set per optimiser family so a search is repeatable. Results are written under the
output directory named in the configuration, with one row per trial recording the sampled
hyperparameters and the resulting objective, plus a record of the best configuration found. The
RC, PERC, and NGRC searches support resuming: re-running with the same output directory
continues from the trials already on disk rather than repeating them.

The example searches write to `outputs/optimizer_search/`, which is ignored by version control.
The best configuration a search finds can be handed to the matching runner
(`run_ngrc.py`, `run_rc.py`, `run_perc.py`, `run_pinn.py`) for a full single-model evaluation.

## Selection and the winner's curse

Picking the single best-scoring trial from a search overstates performance, because the best
score is partly luck — especially for the stochastic models, where a lucky random seed can
inflate a score that a typical seed would not reproduce. The thesis addressed this by
re-ranking the top candidates over 15 fresh seeds and selecting the candidate with the best
median validation horizon, then reporting it at a representative (median) seed rather than the
best one.

This correction consistently lowered the headline numbers. For example, one PINN latent-8
configuration scored 28.8 during its search but had a seed-robust median validation horizon of
10.4. The lesson carried by these scripts is that a search score is a selection signal, not a
reported result: the value to report comes from a separate, seed-aware evaluation. The
per-model selection stories, including which candidate won and by how much, are in the result
cards under
[`final_thesis_package_v001/reports/result_cards/`](../final_thesis_package_v001/reports/result_cards/).

## What is and is not reproducible here

The example searches run against the shipped KS data and are small demonstrations of the
interface. They are not scientifically sufficient searches and are not expected to reproduce the
thesis hyperparameters. The larger historical searches, their databases, and their logs are
intentionally excluded, so the thesis *selections* cannot be re-run from this repository — only
inspected through the shipped configurations, results, and result cards.

The 24 optimiser-search *figures*, however, do regenerate from public files: a compact, sanitised
dataset of the 32 authoritative trial histories (3,040 trials) is shipped under
[`../forecasting/data/optimizer_search/`](../forecasting/data/optimizer_search/), and the figures
are rebuilt with `forecasting/scripts/plot_forecasting_visuals.py` and the public plot configs
`forecasting/configs/03_plots/optimizer_search_*_public.json`. The red star marks the search-time
metric optimum, which equals the finally selected model only for the two deterministic NGRC cases;
the other six finals were chosen by seed-robustness reranking or deterministic repair. Identical
cross-machine PNG checksums are not guaranteed.
