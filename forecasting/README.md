# Forecasting

This module holds the forecasting models compared in the study and the tools to run, tune, and
plot them. For the scientific overview see [../docs/forecasting.md](../docs/forecasting.md).

## Models and settings

Four models are provided:

- **NGRC** — Next-Generation Reservoir Computing (deterministic).
- **RC** — Reservoir Computing (stochastic reservoir).
- **PERC** — Physics-Enhanced Reservoir Computing.
- **PINN** — Physics-Informed Neural Network.

Each runs in two settings, selected by `data_mode` in its configuration:

- `full64` — the model predicts the 64-dimensional physical KS field directly.
- `latent8` — the model predicts in the AE-SINDy latent-8 coordinates, and the latent
  predictions are decoded back to the physical field before evaluation.

All final comparisons are made in the physical 64-dimensional space, so latent predictions are
always decoded before their metrics are computed.

## Code layout

- `src/forecasting/` — the importable package: `config`, `data`, `decoding`, `metrics`,
  `preprocessing`, and `registry` helpers; `models/` (one model class per method); `runners/`
  (one orchestrator per method that loads data, fits and evaluates, and writes results); and
  `visualization/` (figure generation).
- `scripts/` — the command-line entry points:
  - `run_ngrc.py`, `run_rc.py`, `run_perc.py`, `run_pinn.py` — evaluate a single configuration.
  - `optimizer_search_ngrc.py`, `optimizer_search_rc.py`, `optimizer_search_perc.py`,
    `optimizer_search_pinn.py` — tune a model by Bayesian optimisation.
  - `plot_forecasting_visuals.py` — regenerate the per-model forecasting figures.
- `examples/optimizer_search/` — small, runnable optimiser-search configurations; see that
  directory's README.

The package is imported by adding `forecasting/src` to the Python path. The reproduction wrapper
does this automatically; when running a script directly, set `PYTHONPATH=forecasting/src` if the
script does not add it itself (the optimiser searches for NGRC, RC, and PERC add it themselves;
the PINN search expects it to be set).

## Input data

Runs read the shipped KS data. For `full64`, that is the physical trajectory at
`final_thesis_package_v001/results/00_simulation/u_series.npy`. For `latent8`, runs additionally
use the frozen decoder and exported latent arrays under
`final_thesis_package_v001/results/01_ae_sindy/latent8_trial014_representation/`. Decoding between
latent and physical space uses `scripts/ae_sindy_latent_codec.py` in the top-level `scripts/`
directory (one level above this module).

The locked configurations for the eight thesis models are in
`../final_thesis_package_v001/configs/02_forecasting/`, and their results are in
`../final_thesis_package_v001/results/02_forecasting/`. Run scripts from the repository root, for
example:

```bash
python forecasting/scripts/run_ngrc.py --config final_thesis_package_v001/configs/02_forecasting/ngrc_full64.json
```

## Outputs

Single runs and optimiser searches write under the output directory named in the configuration.
The example searches use `outputs/optimizer_search/`, which is ignored by version control. Runs
never write into the shipped package.

## Reproduction versus tuning on new data

There are two different activities here, and it helps to keep them apart. Reproducing the thesis
runs the shipped locked configurations and reproduces the recorded numbers. Tuning searches for
good hyperparameters on data you provide; the example searches show how, on the shipped data, as
an interface demonstration rather than a scientific search.

The thesis hyperparameters came from much larger historical searches. Those searches, their
databases, and their logs are not part of this repository, so the thesis selections can be
inspected through the shipped configurations and result cards but not regenerated here.

## Figures

`plot_forecasting_visuals.py` regenerates the 32 per-model forecasting figures from accepted
result directories, using the plot configuration in
`../final_thesis_package_v001/configs/03_plots/forecasting_visuals.json`. The 24 optimiser-search figures
regenerate with the same script from the compact public optimiser dataset
(`data/optimizer_search/`, 32 trial histories / 3,040 trials) and the public plot configs
`configs/03_plots/optimizer_search_*_public.json`; the 5 seed-robustness figures regenerate with
`scripts/plot_seed_robustness.py` from the shipped seed-robust provenance CSVs. The raw historical
search trees are not shipped, but this plot-relevant subset is.
