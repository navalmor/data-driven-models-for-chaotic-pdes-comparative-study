# Forecasting

The forecasting study compares four models on the task of predicting the KS dynamics forward
in time. Each model is run in two settings: on the full 64-dimensional physical field
(`full64`) and in the eight-dimensional AE-SINDy latent space (`latent8`), giving eight models
in total.

The code is in [`forecasting/`](../forecasting/); see
[`forecasting/README.md`](../forecasting/README.md) for the package layout.

## The four models

- **NGRC** (Next-Generation Reservoir Computing) builds a nonlinear feature library from
  time-delayed states and fits a linear readout by ridge regression. It is deterministic: it
  has no random component.
- **RC** (Reservoir Computing) drives a fixed random recurrent reservoir and fits a linear
  readout. It is stochastic through the random reservoir.
- **PERC** (Physics-Enhanced Reservoir Computing) is RC with an added physics-based constraint
  derived from the KS spatial structure.
- **PINN** (Physics-Informed Neural Network) trains a neural network with a physics residual
  term in its loss.

## Physical and latent settings

In the `full64` setting a model sees the physical field directly. In the `latent8` setting it
operates on the AE-SINDy latent trajectory: predictions are made in latent coordinates and
decoded back to the physical field for evaluation, using the frozen decoder shipped with the
[latent-8 representation](ae_sindy.md). Because the latent models never see the full field,
their accuracy is bounded by the representation as well as by the forecaster. The clearest
illustration is NGRC, whose test horizon drops from 324.0 on the full field to 9.8 in latent
coordinates.

## Data handling

All models use the same frozen trajectory and the same split: the first 20000 steps for
training, the next 5000 for validation, and the remainder (to step 30106) held out for testing.
Inputs are standardised before modelling. The exact splits and preprocessing are recorded in
each model's configuration.

## Running the models

Each model has a runner that takes a single configuration file:

```bash
python forecasting/scripts/run_ngrc.py --config final_thesis_package_v001/configs/02_forecasting/ngrc_full64.json
python forecasting/scripts/run_rc.py   --config final_thesis_package_v001/configs/02_forecasting/rc_full64.json
python forecasting/scripts/run_perc.py --config final_thesis_package_v001/configs/02_forecasting/perc_full64.json
python forecasting/scripts/run_pinn.py --config final_thesis_package_v001/configs/02_forecasting/pinn_full64.json
```

Run from the repository root. The locked configurations for all eight models are in
[`final_thesis_package_v001/configs/02_forecasting/`](../final_thesis_package_v001/configs/02_forecasting/),
and the corresponding results are in
[`final_thesis_package_v001/results/02_forecasting/`](../final_thesis_package_v001/results/02_forecasting/).
The reproduction wrapper runs all eight as part of the default workflow; see
[validation.md](validation.md).

Each runner accepts the shared logging options `--verbose`, `--debug`,
`--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}` and `--log-file PATH`, and writes its
structured logs to stderr. At the default INFO level a run reports its configuration, the
complete input trajectory (shape, representation and dtype of the whole loaded series), the
device, its stages and its primary metrics; the train/validation/test split shapes and the
internal feature and reservoir-state shapes are DEBUG detail. Horizons are logged in physical
simulation time, matching the `validation_horizon_physical` and `test_horizon_physical` columns
of [`forecasting_final_metrics.csv`](../final_thesis_package_v001/tables/forecasting_final_metrics.csv);
the underlying step count is available at DEBUG. PINN reports periodic epoch progress (roughly
ten updates by default, every epoch under `--verbose`/`--debug`) and warns once if automatic
device selection falls back to CPU. Logging affects only what is printed, never the numbers. The
shared format and behaviour are documented once in the [README](../README.md#logging-and-verbosity).

## Selection versus final evaluation

The reported numbers separate model selection from final evaluation. Hyperparameters were
chosen using the validation split only. For the stochastic models (RC, PERC, PINN) selection
used a seed-robust rule rather than the single best run; this is described in
[optimizer_search.md](optimizer_search.md). The test split was evaluated once, after the
configuration was locked. The consequences for how the numbers should be read are summarised in
[results_summary.md](results_summary.md).

## Plotting

`forecasting/scripts/plot_forecasting_visuals.py` regenerates the per-model forecasting figures
(32 figures: validation and test, relative-L2 horizon and spatiotemporal comparison, for each
of the eight models) from accepted result directories. The plot configuration is
[`final_thesis_package_v001/configs/03_plots/forecasting_visuals.json`](../final_thesis_package_v001/configs/03_plots/forecasting_visuals.json).

The forecasting per-model figures use a Lyapunov-time axis, while the tables report physical
simulation time, so a horizon read off a figure axis will not equal the table value. Quote the
tables, not the axes.

### Rollout triptych colour scales

Every true / prediction / error three-panel figure in this repository follows one convention,
implemented once in `common.plotting` (`truth_prediction_limits`, `error_limits`) and used by
every triptych generator:

> Panels (a) and (b) share a color scale determined from the true field, enabling direct
> comparison; predicted values outside this range are visually saturated. Panel (c) uses an
> independent zero-centered error scale.

Concretely, the shared field range is `±nanmax(abs(u_true))` taken from the displayed true
field alone, and the error range is `±nanmax(abs(u_true − u_pred))`. The prediction never
influences the field range, and no percentile widening or narrowing is applied.

This matters for diverging rollouts. If the prediction were allowed to set the scale, a model
whose forecast leaves the physical amplitude range would compress the true field into a
featureless band and hide the structure the figure exists to show. Instead the prediction
saturates, and that saturation is itself the signal: it marks where the forecast has left the
amplitude range represented by the truth. Only the display normalisation saturates — the
underlying prediction arrays are never clipped or modified.
