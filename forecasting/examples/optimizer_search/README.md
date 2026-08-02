# Optimiser-search examples

These four configurations show how to run each model's optimiser search. They are interface
demonstrations, not scientific searches: the budgets are deliberately tiny and the results are
not expected to reproduce the thesis hyperparameters. For the method and its limitations, see
[../../../docs/optimizer_search.md](../../../docs/optimizer_search.md).

Each example runs against the shipped KS trajectory, so no external data is needed.

## Running the examples

Run from the repository root. The NGRC, RC, and PERC searches add the package to the Python path
themselves; the PINN search needs it set explicitly.

```bash
python forecasting/scripts/optimizer_search_ngrc.py --config forecasting/examples/optimizer_search/ngrc_example.json
python forecasting/scripts/optimizer_search_rc.py   --config forecasting/examples/optimizer_search/rc_example.json
python forecasting/scripts/optimizer_search_perc.py --config forecasting/examples/optimizer_search/perc_example.json
PYTHONPATH=forecasting/src python forecasting/scripts/optimizer_search_pinn.py --config forecasting/examples/optimizer_search/pinn_example.json
```

The PINN example trains a small neural network for each trial and is noticeably more expensive
than the other three, even at this reduced budget. NGRC, RC, and PERC run on CPU with NumPy and
SciPy; PINN needs PyTorch.

## Input data

All four point `data` at the shipped physical trajectory:

```
final_thesis_package_v001/results/00_simulation/u_series.npy
```

This is a float64 NumPy array of shape `(time_steps, 64)`. The `train_end_index` and
`valid_end_index` fields split it into training, validation, and test ranges by index; the
searches use the validation range only. NGRC, RC, and PERC read the key `data.physical_data_path`;
PINN reads `data.data_path`.

## Using your own data

To search on a different dataset, change the data path and the split indices to match your array,
and keep the array shape `(time_steps, native_dimension)`. For the full-field setting the native
dimension is 64. The examples all use `data_mode: full64`; the latent setting additionally needs
the exported latent data and decoder, which is why the examples stay on the physical field.

## Budgets

The budgets are intentionally small so the examples finish quickly:

- NGRC, RC, PERC: 8 trials per optimiser family, 4 initial points.
- PINN: 4 trials with the random (`dummy`) optimiser and a reduced number of epochs.

A real search uses far larger budgets and more optimiser families. These values are for checking
that the pipeline runs end to end, not for finding good hyperparameters.

## Outputs

Each search writes under the directory named in `output.base_dir`, which the examples set to
`outputs/optimizer_search/` — a location ignored by version control. A search records one row per
trial with the sampled hyperparameters and the resulting validation objective, and it records the
best configuration found. That best configuration can be handed to the matching runner
(`run_ngrc.py`, `run_rc.py`, `run_perc.py`, `run_pinn.py`) for a full single-model evaluation.

The RC, PERC, and NGRC searches can be resumed: re-running with the same output directory
continues from the trials already on disk. The searches never write into the shipped package.

## Relation to the thesis searches

The thesis hyperparameters came from much larger historical searches whose raw output is not part
of this repository. These examples cannot regenerate those selections; they show how the same
tools are driven. The shipped optimiser-search figures, however, can be regenerated from the
compact public trial-history dataset under `../../data/optimizer_search/` using the public
plotting configurations in `../../configs/03_plots/optimizer_search_*_public.json`; the full
private multi-gigabyte search archive is not required. The highlighted (red star) point denotes
the search-time metric optimum. It is also the final selected model only for the two NGRC cases
(`ngrc_full64`, `ngrc_latent8`); the RC, PERC, and PINN models were selected afterwards by
seed-robustness reranking or deterministic repair. Scientific and visual regeneration are
supported, though identical cross-machine PNG checksums are not guaranteed.
