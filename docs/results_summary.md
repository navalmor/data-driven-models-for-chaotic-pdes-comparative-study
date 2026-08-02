# Results summary

This is a compact overview of the final results. The authoritative numbers are in
[`final_thesis_package_v001/tables/`](../final_thesis_package_v001/tables/), and the full
development and selection history for each result is in the twelve result cards under
[`final_thesis_package_v001/reports/result_cards/`](../final_thesis_package_v001/reports/result_cards/).
All horizons below are in physical simulation time.

## How to read these numbers

Two points matter for reading the forecasting results honestly:

- **Selection used the validation split only, and the test split was evaluated once** after each
  configuration was locked. Test numbers are genuine held-out results, not tuning targets.
- **The stochastic models were corrected for the winner's curse.** For RC, PERC, and PINN, the
  single best-scoring search result overstates performance, because it is partly a lucky random
  seed. Final selection used the best median validation horizon over 15 fresh seeds, reported at
  a representative seed. This lowered the reported numbers relative to the originally published
  ones, which is what makes them defensible. Details are in
  [optimizer_search.md](optimizer_search.md).

## AE-SINDy

From [`tables/ae_sindy_final_metrics.csv`](../final_thesis_package_v001/tables/ae_sindy_final_metrics.csv):

- **Latent-8 representation** — the coordinate system for the latent forecasting models.
  Reconstruction MSE about 0.00659, MAE about 0.0620. Selected for reconstruction quality, not
  as a dynamics model.
- **Latent-6 dynamics baseline** — the interpretable contribution: 17 active quadratic terms,
  test rollout horizon about 8.8. Shipped as a frozen-checkpoint reanalysis.
- **Post-hoc predictive refit** (appendix) — mean test horizon about 10.86, but dense (about 81
  percent of coefficients active), so predictive rather than interpretable.

## Forecasting

From [`tables/forecasting_final_metrics.csv`](../final_thesis_package_v001/tables/forecasting_final_metrics.csv):

| model | representation | validation horizon | test horizon |
|---|---|---|---|
| NGRC | full64 | 379.9 | 324.0 |
| NGRC | latent8 | 24.2 | 9.8 |
| RC | full64 | 24.4 | 15.9 |
| RC | latent8 | 20.0 | 16.6 |
| PERC | full64 | 25.6 | 39.3 |
| PERC | latent8 | 16.2 | 3.8 |
| PINN | full64 | 19.8 | 7.3 |
| PINN | latent8 | 10.4 | 8.4 |

Two findings stand out:

- **NGRC on the full field is the strongest model by a wide margin**, roughly fifteen times the
  horizon of any other. It is deterministic, so its search is its selection procedure. Its number
  was corrected downward, from a superseded 419.2 to 379.9, after an earlier configuration was
  found to rely on an ill-conditioned, near-zero ridge that made the result depend on
  floating-point details. The lower value is the reproducible one.
- **Latent-space forecasting is bounded by the representation.** The same NGRC method drops from a
  test horizon of 324.0 on the full field to 9.8 in latent coordinates.

The stochastic corrections were substantial. PERC full64 showed the widest seed spread in the
study, with a lucky seed reaching 64.3 against a median of 25.6; PINN latent8 was corrected from
a published 26.7 to 10.4.

**PINN full64 was corrected for a different reason.** Its published configuration carried a
physics weight of exactly zero, so the physics residual was never computed and the model was
physics-informed in name only. The result shown above comes from a rebuilt configuration with a
genuinely active physics term, selected on validation evidence alone by the highest median
validation horizon across fifteen seeds, and locked before any test data was read. Its validation
median is 19.8 with a wide seed spread (8.5 to 23.1), and its single test evaluation returned
7.3 — well below validation. That drop is reported as found; it did not prompt any retuning,
because changing the configuration after seeing test data would destroy the validation-only
guarantee. A paired ablation on this one configuration, varying only the physics weight, favours
the active setting on validation (19.8 against 14.8, better on 11 of 15 seeds). No equivalent
comparison exists on the test split, so this result does not show that physics improved test
performance, and it is not evidence that physics-informed training is generally superior. The
old and new PINN full64 configurations differ in architecture, hyperparameters, seed and
selection procedure, so their test numbers do not isolate the effect of the physics term.

## Frozen versus reproduced

Not everything is regenerated on each run. The simulation dataset and the latent-8 autoencoder
representation are frozen and validated by checksum, for the reasons given in
[simulation.md](simulation.md) and [ae_sindy.md](ae_sindy.md); the latent-6 baseline ships as a
frozen-checkpoint reanalysis. The remaining results — the post-hoc refit and all eight
forecasting models — are reproduced from the frozen inputs. What the default workflow checks, and
the exact counts, are in [validation.md](validation.md).

## Limitations

- Horizons are chaotic-system quantities and are short and window-variable in latent space.
- The stochastic models carry seed spread; report them with their distributions, not single
  numbers.
- The originally published pre-correction numbers are superseded and should not be quoted.
- The larger historical hyperparameter searches are not included, so the selections can be
  inspected through the shipped artifacts but not regenerated.
