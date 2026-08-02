# Optimized Data-Driven Models for PDEs: A Comparative Study

This repository accompanies a master's thesis comparing data-driven models for forecasting a
chaotic partial differential equation. It contains the code, the frozen results, and a
reproduction and validation workflow, so the results can be inspected and checked rather than
taken on trust.

## The problem

The test system is the one-dimensional Kuramoto-Sivashinsky (KS) equation, a fourth-order
nonlinear PDE whose solutions become spatiotemporally chaotic. Chaos makes it a demanding
forecasting target: small differences grow quickly, so a model's useful prediction horizon is
limited and sensitive to how it is built and selected. A single frozen KS trajectory is the
common reference for every result here; see [docs/simulation.md](docs/simulation.md).

## The models

The study covers a reduced-order modelling method and four forecasting models:

- **AE-SINDy** — an autoencoder paired with sparse dynamics identification, used both to learn a
  low-dimensional latent representation and to discover interpretable latent dynamics.
- **NGRC** — Next-Generation Reservoir Computing.
- **RC** — Reservoir Computing.
- **PERC** — Physics-Enhanced Reservoir Computing.
- **PINN** — Physics-Informed Neural Network.

Each forecasting model is studied on the full 64-dimensional field and in the eight-dimensional
AE-SINDy latent space. See [docs/ae_sindy.md](docs/ae_sindy.md) and
[docs/forecasting.md](docs/forecasting.md).

## What you can do with this repository

- Read the model implementations and the exact configurations that produced the results.
- Validate the shipped package against its recorded contract.
- Run the default reproduction, which reruns the reproducible results on CPU.
- Regenerate the final forecasting figures.
- Run the optimiser-search examples on compatible data.
- Inspect the optional GPU retraining workflow for the autoencoder.

## Repository map

| path | contents |
|---|---|
| `final_thesis_package_v001/` | the curated results package: configs, results, plots, tables, manifests, and result cards |
| `repro_validation/` | the validation contract (expected values, tolerances, checksums) |
| `scripts/` | the reproduction wrapper and supporting scripts |
| `ae_sindy/` | the AE-SINDy implementation and its runners |
| `forecasting/` | the forecasting models, runners, optimiser searches, plotting, and examples |
| `simulation/` | the KS solver |
| `docs/` | topic guides: [environment](docs/environment.md), [simulation](docs/simulation.md), [AE-SINDy](docs/ae_sindy.md), [forecasting](docs/forecasting.md), [optimiser search](docs/optimizer_search.md), [validation](docs/validation.md), [horizon quantities](docs/horizon_quantities.md), [advanced GPU retraining](docs/advanced_latent8_gpu_retrain.md), [results summary](docs/results_summary.md) |
| `tests/` | the test suite: style contract, plotting conventions, public dataset, release scope, reproduction entry point |
| `common/` | shared plotting style contract and helpers |
| `outputs/`, `reproduction_runs/` | generated output only, never tracked; safe to delete |

## Installation

### Git LFS is required

The frozen trajectory, the model checkpoints and all 76 figures are stored with
[Git LFS](https://git-lfs.com). A clone made without it yields ~130-byte pointer files in
their place, and every checksum check then fails. Install LFS and fetch the content **before
anything else**:

```bash
git lfs install
git lfs pull
```

The reproduction wrapper detects unresolved pointers and stops with this instruction rather
than failing obscurely.

### Python environment

The default workflow runs on CPU and needs only the pinned dependencies in
[`requirements.txt`](requirements.txt), on Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the install:

```bash
python -c "import numpy, scipy, sklearn, skopt, matplotlib, torch; \
print(numpy.__version__, scipy.__version__, sklearn.__version__, \
matplotlib.__version__, skopt.__version__, torch.__version__)"
```

The exact environment used for the GPU work, including the CUDA stack, is recorded in
[`environment-lock.yml`](environment-lock.yml). PyTorch installed from PyPI differs from the
Conda build used for the packaged checkpoints; the details and their consequences are in
[docs/environment.md](docs/environment.md).

Some package configurations record `experiments/...` paths in fields such as
`selection_input_config` and `optimizer_search_directory`. These name the historical search
directories a configuration came from. They are provenance metadata, not runtime inputs: every
path the code reads resolves inside this repository, and the historical search trees are not
shipped.

## Three supported routes

The routes differ in what they *do*, not only in how long they take. Only the third reruns
scientific computation.

| route | what it does | runs models? | time |
|---|---|---|---|
| **1. Verify** | checks the frozen package against the validation contract | **no** | seconds |
| **2. Regenerate plots** | redraws the 76 figures from the frozen data | **no** | minutes |
| **3. Reproduce** | retrains and re-evaluates the locked configurations | **yes** | ~30 min |

### Route 1 — verify the frozen package (start here)

```bash
python scripts/reproduce_final_thesis.py --validate-existing --yes
```

It runs no models and prints:

```
54 contract targets checked
  PASS 52 · FAIL 0 · REVIEW_REQUIRED 0 · SKIPPED 2 · ERROR 0
Verdict    : PASSED  (exit 0)
```

The two skipped targets are the seed checks for the deterministic NGRC models, which have no
random seed. Exit codes: `0` pass, `1` failure, `3` review required.

### Route 2 — regenerate the figures from frozen data

Every command below redraws figures from stored arrays and results. No solver, training or
search is involved. Run them from the repository root; together they produce all **76**
package figures under `outputs/figures/`, which is untracked, so the package itself is never
overwritten.

Each `plot_forecasting_visuals.py` run needs an output directory that does not yet exist. The
configs set `output.overwrite = false`, so pointing two runs at the same root fails with
`Output directory exists and overwrite=false`. The distinct roots below avoid that.

**Forecasting and optimiser-search figures — 56**

```bash
python forecasting/scripts/plot_forecasting_visuals.py \
    --config final_thesis_package_v001/configs/03_plots/forecasting_visuals.json \
    --output-root outputs/figures/forecasting            # 32 per-model figures

for family in rc ngrc perc pinn; do
    python forecasting/scripts/plot_forecasting_visuals.py \
        --config final_thesis_package_v001/configs/03_plots/optimizer_search_${family}.json \
        --output-root outputs/figures/optimizer_${family} # 6 per family, 24 in total
done
```

**Seed-robustness and physics-ablation figures — 7**

```bash
python forecasting/scripts/plot_seed_robustness.py \
    --input-dir final_thesis_package_v001/results/02_forecasting/provenance/seed_robust \
    --output-dir outputs/figures/comparison/seed_robustness   # 6 figures

python forecasting/scripts/plot_physics_ablation.py \
    --output-base outputs/figures/comparison/pinn_full64_physics_ablation/paired_active_vs_zero_weight
                                                              # 1 figure
```

**Spatiotemporal simulation figure — 1**

The frozen trajectory is loaded from the package and passed straight to the plotting helper,
so the solver never runs:

```bash
python - <<'PY'
import numpy as np
from simulation.kse_simulation import save_spatiotemporal_plot

frozen = "final_thesis_package_v001/results/00_simulation"
u = np.load(f"{frozen}/u_series.npy")
x = np.load(f"{frozen}/x.npy")
t = np.load(f"{frozen}/t.npy")
save_spatiotemporal_plot(u, x, t, "outputs/figures/00_simulation",
                         use_lyapunov=True, lambda_max=0.043)
PY
```

**AE-SINDy figures — 12**

The two selected runs are visualised from their stored `metrics.pkl`, `analysis_summary.pkl`
and `validation_losses.npy`. Nothing is trained and no forward pass is performed. The shipped
configs leave `visualization_output_dir` unset, which would place figures inside the package,
so the first step writes copies that redirect the figures; `--log-file` redirects the log for
the same reason. Both copies keep `output_dir` unchanged, so each run directory is still read
from its frozen location:

```bash
python - <<'PY'
import json, pathlib

runs = {
    "latent8_trial014_representation":
        "final_thesis_package_v001/configs/01_ae_sindy/latent8_trial014_representation.json",
    "latent6_trial002_dynamics_baseline":
        "final_thesis_package_v001/configs/01_ae_sindy/"
        "latent6_trial002_dynamics_baseline_checkpoint_reanalysis.json",
}
pathlib.Path("outputs/configs").mkdir(parents=True, exist_ok=True)
for name, source in runs.items():
    config = json.loads(pathlib.Path(source).read_text())
    config["visualization_output_dir"] = f"outputs/figures/01_ae_sindy/{name}"
    pathlib.Path(f"outputs/configs/{name}_visualize.json").write_text(
        json.dumps(config, indent=2))
PY

for run in latent8_trial014_representation latent6_trial002_dynamics_baseline; do
    python ae_sindy/scripts/run_visualize.py \
        --config outputs/configs/${run}_visualize.json \
        --log-file outputs/logs/${run}_visualize.log     # 6 figures per run
done
```

**Where each group lands, and how to check the total**

| command | figures | output location |
|---|---|---|
| `plot_forecasting_visuals.py` (forecasting visuals) | 32 | `outputs/figures/forecasting/` |
| `plot_forecasting_visuals.py` (4 optimiser families) | 24 | `outputs/figures/optimizer_<family>/` |
| `plot_seed_robustness.py` | 6 | `outputs/figures/comparison/seed_robustness/` |
| `plot_physics_ablation.py` | 1 | `outputs/figures/comparison/pinn_full64_physics_ablation/` |
| `save_spatiotemporal_plot` | 1 | `outputs/figures/00_simulation/` |
| `run_visualize.py` (2 runs) | 12 | `outputs/figures/01_ae_sindy/<run>/` |
| | **76** | |

```bash
find outputs/figures -name '*.png' | wc -l     # expect 76
git status --porcelain                         # expect no output
```

The second check is the one that matters: `outputs/` is ignored by Git, so a clean status
confirms that no packaged figure, result, config or manifest was modified. Figures are written
at 300 dpi. Identical cross-machine PNG checksums are not guaranteed, so compare figures
visually rather than by hash.

### Route 3 — reproduce the scientific results

This **retrains and re-evaluates** the locked configurations. It never searches, never
reselects a candidate or seed, and never rewrites the package: output goes to
`reproduction_runs/`, which is untracked.

```bash
python scripts/reproduce_final_thesis.py --list            # the 12 final results
python scripts/reproduce_final_thesis.py --plan --results ngrc_full64   # preview, creates nothing
python scripts/reproduce_final_thesis.py --results ngrc_full64 --yes    # one result
python scripts/reproduce_final_thesis.py --all-results --yes            # all 12
```

The full set takes about 30 minutes on CPU, of which `pinn_full64` is roughly 21, and writes
about 340 MB. `--all-safe` selects the ten laptop-safe results, excluding the two heavy PINNs.

A fresh run validates fewer targets than route 1: `PLOT-001` and `TBL-001` are package-wide
checks that a per-result run does not evaluate. Details in
[docs/validation.md](docs/validation.md).

## Logging and verbosity

Every entry point — the reproduction wrapper, the forecasting runners, the optimiser searches,
the AE-SINDy tools, the simulation and the plotting scripts — shares one logging format:

```
HH:MM:SS | LEVEL   | COMPONENT  | message
```

The two output streams are kept strictly apart. **All** structured operational logs go to
stderr, at every level; **stdout carries only plain presentation** — the interactive menu, input
prompts, `--list` and `--plan` text, workflow headings, tables and the formal validation and
reproduction summaries, printed without a log prefix and unaffected by the log level. A run can
therefore be redirected cleanly: `> results.txt` keeps the summaries, `2> run.log` keeps the
diagnostics.

At the default INFO level each component reports its start, the **complete input trajectory**
(shape, representation and dtype of the whole loaded series, not of one split), the device for
the PyTorch models, its major stages, its existing primary metrics (tagged `METRIC |`), the
output location and the elapsed time. Train/validation/test split shapes and internal shapes —
feature matrices, reservoir states, batch tensors — are DEBUG detail. Forecast horizons are
reported in physical simulation time, the same quantity the tables record as
`validation_horizon_physical`; the corresponding step count is available at DEBUG. Logging never
changes any scientific result.

The default mode and four logging options are available on every entry point (the four
options are mutually exclusive except `--log-file`):

| option | effect |
|---|---|
| *(default)* | INFO: concise progress, periodic iterative updates, primary metrics, short errors |
| `--verbose` | more frequent INFO-level progress, including per-epoch or per-trial updates where applicable |
| `--debug` | DEBUG diagnostics: resolved interpreters and commands, seeds, deterministic settings, internal shapes, return codes and full tracebacks |
| `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}` | set the level explicitly (case-insensitive) |
| `--log-file PATH` | additionally write a persistent, full-detail UTF-8 log (append) |

A persistent log file always records full DEBUG detail regardless of the terminal level, and
uses a more detailed line format than the terminal: a full `YYYY-MM-DD HH:MM:SS` date plus the
logger name and source line number, before the usual level, component and message.

By default the PINN training loop reports roughly ten evenly spaced progress updates (always
including the first and last epoch) rather than one line per epoch; `--verbose`/`--debug` report
every epoch. When device selection is automatic and no GPU is present, the fall back to CPU is
reported once as a warning — CPU is the validated path for this package, not an error. Expected
input or configuration failures print a single concise error; the full traceback is shown only
under `--debug`.

```bash
python scripts/reproduce_final_thesis.py --all-forecasting --yes --verbose

python scripts/reproduce_final_thesis.py --results pinn_latent8 --yes --debug \
  --log-file reproduction_runs/pinn_latent8_debug.log
```

When the wrapper runs the individual models as child processes, each child inherits the terminal,
so its component logs appear live on stderr underneath the workflow heading as the model runs. The
same records are also written to a per-component file under `reproduction_runs/<run-id>/logs/`,
which the wrapper hands to the child as its own `--log-file`; no two processes ever share a file.
An explicitly selected level or verbosity is propagated to the children as well.

## Frozen results and fresh reproduction

Two inputs are frozen and validated by checksum rather than regenerated:

- The KS trajectory is chaotic and cannot be reproduced bit for bit across environments, so the
  solver is not re-run by default.
- The GPU-trained latent-8 autoencoder representation is validated by checksum; everything
  downstream reproduces from the exported latents on CPU.

Everything else — the post-hoc refit and all eight forecasting models — is reproduced from these
frozen inputs. See [docs/simulation.md](docs/simulation.md) and [docs/ae_sindy.md](docs/ae_sindy.md).

## Figures

All 76 package figures regenerate from public-repository files. [Route 2](#route-2--regenerate-the-figures-from-frozen-data)
gives the complete command for every group; this section records where the inputs come from.

The 32 per-model forecasting figures and the 24 optimiser-search figures both come from
`forecasting/scripts/plot_forecasting_visuals.py`. The optimiser figures draw on the compact
public optimiser dataset (`forecasting/data/optimizer_search/`, 32 trial histories / 3,040
trials) and the public plot configs in
`forecasting/configs/03_plots/optimizer_search_*_public.json`; the raw ~34 GB search trees are
not shipped, but this plot-relevant subset is. The 6 seed-robustness figures come from
`forecasting/scripts/plot_seed_robustness.py` and the shipped seed-robust provenance CSVs, and
the physics-ablation figure from `forecasting/scripts/plot_physics_ablation.py`.

The remaining 13 are drawn from frozen arrays and stored run artefacts rather than from a
plot config. The spatiotemporal figure calls `save_spatiotemporal_plot` in
`simulation/kse_simulation.py` on the stored trajectory, so the solver is not involved. The 12
AE-SINDy figures come from `ae_sindy/scripts/run_visualize.py`, which reads the metrics,
analysis summary and validation losses already stored in each selected run directory; no model
is trained or evaluated. Identical cross-machine PNG checksums are not guaranteed.

## Optimiser searches

Each forecasting model has an optimiser-search script, with runnable examples in
[`forecasting/examples/optimizer_search/`](forecasting/examples/optimizer_search/). The examples
demonstrate the interface on the shipped data; they are not full searches and do not reproduce the
thesis hyperparameters. See [docs/optimizer_search.md](docs/optimizer_search.md).

## Results at a glance

The headline result is deterministic NGRC on the full field, which forecasts substantially
further than any other model. The stochastic models were corrected for the winner's curse by
seed-robust selection, which lowered their reported numbers. The compact summary is in
[docs/results_summary.md](docs/results_summary.md), and the authoritative numbers are in
[`final_thesis_package_v001/tables/`](final_thesis_package_v001/tables/).

## Advanced GPU workflow

An optional, non-default workflow retrains the latent-8 autoencoder on a GPU to corroborate the
frozen representation. It is hardware-specific and is not required by the default reproduction;
see [docs/advanced_latent8_gpu_retrain.md](docs/advanced_latent8_gpu_retrain.md).

## Licensing

This repository uses two licences:

- **Source code, scripts, and configuration logic:** MIT, in [`LICENSE`](LICENSE).
- **Data, model checkpoints, trained artefacts, prediction arrays, the 76 figures, tables and
  result data:** Creative Commons Attribution 4.0 International (CC BY 4.0), in
  [`LICENSES/CC-BY-4.0.txt`](LICENSES/CC-BY-4.0.txt).

The AE-SINDy method is that of Champion et al. (2019). The implementation in
[`ae_sindy/`](ae_sindy/) is a PyTorch reimplementation of the public TensorFlow reference
implementation `kpchamp/SindyAutoencoders` released with that paper, adapted here to the KS
data pipeline, normalisation, analysis, rollout, post-hoc refit, and diagnostics. The upstream
project is MIT-licensed; its notice is retained in
[`LICENSES/MIT-SindyAutoencoders.txt`](LICENSES/MIT-SindyAutoencoders.txt). The exact upstream
revision is not recorded in this repository's history.

## Citing

Citation metadata is in [`CITATION.cff`](CITATION.cff). The repository is available at
<https://github.com/navalmor/data-driven-models-for-chaotic-pdes-comparative-study>. No release
identifier or DOI has been assigned yet; cite the commit hash you used.

## Author and acknowledgement

This repository accompanies the Master's thesis of **Naval Shukhabhai Mor**, Department of Data
Science, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).

The work was supervised by **Prof. Dr. Marius Yamakou**, whose guidance shaped the study's
direction and scope. The supervision is gratefully acknowledged; responsibility for the code,
results and any remaining errors rests with the author.

## Limitations

- The KS trajectory and the latent-8 autoencoder representation are validated by checksum, not
  regenerated; the reproduction contract passes within stated numeric tolerances and does not
  promise bitwise-identical output on arbitrary hardware.
- The stochastic models carry seed spread and should be reported with their distributions.
- The originally published pre-correction numbers are superseded and should not be quoted.
- The historical hyperparameter searches are not included; selections can be inspected but not
  regenerated.

## Status

This is a thesis reproducibility and research-code release: a fixed set of results with the code
and the contract needed to check them.
