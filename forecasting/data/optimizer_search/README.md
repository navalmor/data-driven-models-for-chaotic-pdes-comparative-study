# Compact optimiser-search dataset

This directory holds the optimiser trial histories required to reproduce the
twenty-four optimiser figures of the thesis: eight convergence plots, eight
sampling-matrix plots and eight surrogate-landscape plots.

## Purpose

The optimiser figures are the only artifacts that consume the hyperparameter
search histories, and they read a single table per optimiser family. Publishing
those tables makes the figures reproducible from this repository alone. The
per-trial run artifacts of the original searches (model checkpoints, predicted
trajectories, intermediate arrays) amount to tens of gigabytes, are never read by
any figure, and therefore remain private.

## Contents

Eight forecasting cases, each searched with four optimiser families:

| directory | internal name | display name |
|---|---|---|
| `dummy/` | `dummy` | Random |
| `forest/` | `forest` | Random forest |
| `gbrt/` | `gbrt` | GBRT |
| `gp/` | `gp` | Gaussian process |

The cases are `ngrc_full64`, `ngrc_latent8`, `rc_full64`, `rc_latent8`,
`perc_full64`, `perc_latent8`, `pinn_full64` and `pinn_latent8`. Together the
thirty-two tables hold 3040 successful optimiser trials.

```
<case_id>/<optimizer>/optimizer_results.csv
```

Alongside the tables: `schema.json` (column contract), `provenance.json` (source
lineage and checksums), `source_selection_manifest.csv` (one row per approved
history) and `dataset_sha256_manifest.csv` (checksums for every file here).

## Columns

`case_id`, `model` and `representation` identify the forecasting case.
`optimizer` names the search family and always agrees with the containing
directory. `source_run_id`, `source_search_order` and `search_stage` preserve the
original search campaign, which matters for the two cases whose families were
partitioned across more than one search directory: `ngrc_latent8` (random and
Gaussian-process families in one run, tree-based families in another) and
`pinn_full64` (one directory per family). `trial` is the original trial number and
is unique per case and `optimizer`, not globally. `status` is retained so the
plotting filter remains meaningful. `valid_relative_l2_horizon_time` is the
validation-horizon metric that every optimiser figure maximises. The remaining
columns are the search-space parameters declared by the canonical plotting
configuration for that model family.

Values are stored exactly as the searches recorded them. Categorical tokens such
as `full` and `linear_quadratic`, and booleans such as `true` and `false`, are
never normalised: the plotting code orders categories by sorting raw values and
applies its own display mapping. Rewriting them here would silently change the
axis ordering of the published figures.

## The `pinn_full64` physics-active search

The `pinn_full64` trials shipped here come from the v003 physics-active search, in
which `lambda_phy` is a positive search dimension. They replace an earlier search
in which every trial carried `lambda_phy = 0.0`, so the physics residual was never
computed and the case was physics-informed in name only.

Four families of eighty trials give 320 completed trials. Of those, 201 reach the
one-percent final-twenty-percent physics-activity threshold used to judge whether
the physics term contributed meaningfully to training; the remaining 119 ran
correctly but with a physics contribution below that threshold. That eligibility
fact is recorded here rather than as a column, because the schema is shared with
`pinn_latent8` and adding a column would silently change the dataset contract for
the other cases.

## Reproducing the figures

With the pinned environment (`requirements.txt` / `environment-lock.yml`;
`scikit-learn==1.3.2` in particular, which governs the surrogate fit):

```
python forecasting/scripts/plot_forecasting_visuals.py \
  --config forecasting/configs/03_plots/optimizer_search_ngrc_public.json \
  --output-root <preview directory>
```

and likewise for the `rc`, `perc` and `pinn` public configurations. The
surrogate landscape refits a Gaussian process at plot time; because every case
holds fewer rows than the configured `max_train_points`, no subsampling occurs
and the fit is deterministic for a given environment.

## Interpreting the marked optimum

> The red star in optimiser sampling-matrix and surrogate-landscape figures
> denotes the search-time metric optimum. It is the final selected model for the
> two NGRC cases. For the six seed-reranked or deterministically repaired
> forecasting cases, it is diagnostic and does not necessarily denote the final
> published configuration.

The two NGRC cases are deterministic, so the search argmax was adopted directly.
The six remaining cases were finalised differently: `rc_full64`, `rc_latent8`,
`perc_full64`, `perc_latent8`, `pinn_latent8` and `pinn_full64` by re-ranking the
top candidates over fifteen fresh seeds and selecting the highest median
validation horizon. For those cases the optimiser figures remain valid as search
diagnostics, but they are not selection evidence; the seed-robustness figures
serve that role.

For `pinn_full64` the distance between the marked optimum and the published
configuration is large enough to state explicitly. The highest single-seed search
result did not remain the strongest configuration under seed repetition; the final
PINN-64 configuration was selected using median validation horizon across 15
seeds. The red star therefore sits at the search-time optimum, GBRT trial 22 at a
validation horizon of 31.2, while the published configuration is candidate C5,
dummy trial 47, whose fifteen-seed median validation horizon is 19.8. The star is
not moved to C5: it means what it means for every other case in this dataset.

No selected-candidate column is shipped. Doing so would conflate the search-time
optimum, which the plotting code derives from the metric, with the final
post-search selection, which was made by a different procedure.

## Provenance and reproducibility

The trial histories originate from the private research repository at commit
`7d574a44b40f27e40302e9946ef6af5243222cab`, projected by
`forecasting/scripts/extract_optimizer_figure_data.py` against the approved
source-selection audit. `provenance.json` records, for every history, the source
path relative to the private repository root together with the source and
extracted checksums.

This dataset supports scientific reproduction (the same trials, extrema, optima
and surfaces) and visual reproduction (the same figures). Byte-identical PNG
checksums across different machines are not guaranteed: Gaussian-process
hyperparameter fitting and font rasterisation both vary at the last digits with
the numerical libraries and platform in use.
