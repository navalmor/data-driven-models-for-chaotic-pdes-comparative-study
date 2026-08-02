# Result Card — `ngrc_latent8`

## 1. Result identity

**NGRC forecasting in the 8-dimensional AE-SINDy latent space**, decoded back to the physical field for
evaluation. Its value in the thesis is **comparative**: the same deterministic method as `ngrc_full64`, changing only
the space it predicts in.

**Thesis role: MAIN** (as the representation-bound comparison). **Deterministic** — no RNG, no seed-rerank.

## 2. Inputs and dependencies

| item | path |
|---|---|
| package config | [`configs/02_forecasting/ngrc_latent8.json`](../../configs/02_forecasting/ngrc_latent8.json) |
| canonical source config | `latent8_ngrc_selected_v002.json` |
| result | `results/02_forecasting/ngrc/selected/ngrc_latent8_selected_v002/` |
| search source (survives) | `ngrc_latent8_optimizer_search_v002` |

**Depends on:** the package's **own** latent export
`results/01_ae_sindy/latent8_trial014_representation/latent_data/` (`decoder_run_dir` + `native_data_dir`), which in
turn depends on the frozen simulation. The dependency chain closes **inside** the package.

## 3. Development and selection timeline

*Reconstructed from configs, config metadata, reports and on-disk artifacts. Where motivation is explicitly
documented it is cited; where it is inferred, the wording says so; where evidence is absent it says
"unknown from available evidence."*

| Step | Artifact/config/script | Step type | Candidate/search space | Metric or check used | Aim | Outcome / what was learned | Limitation or issue found | Why next step was needed | Connection to final result | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| **Smoke test** | `latent8_ngrc_smoke_v002.json` | smoke | minimal execution, not a search | runner completes; data path resolves; outputs written | Check the runner, config schema and data path execute end-to-end before spending compute. | Pipeline executes for this model/data_mode. Available evidence suggests no scientific claim was drawn from this stage. | A smoke run says nothing about model quality. | A parameter feasibility probe was needed before any search. | Infrastructure only; no bearing on the final numbers. | medium |
| **Probes** | `latent8_ngrc_probe_{bias,delay2,pairwise,trig}_v002.json (4 configs)` | probe | feature-library variants: bias, delay depth 2, pairwise (quadratic), trigonometric - mirroring the full64 probe set | validation relative_l2_horizon_time | Test which NGRC feature-library components matter in the 8-dimensional latent space. | Configuration differences indicate the same four library components were probed as for full64. The specific per-probe outcomes are unknown from available evidence (probe outputs are not retained). | Probes fix structure only. | A grid then optimizer search were needed over continuous parameters. | Fixed the structural choices the later searches held constant. | medium |
| **Grid search v002 (region finding)** | `latent8_ngrc_grid_search_v002.json` | grid_search | coarse grid over NGRC latent8 parameters | validation relative_l2_horizon_time | Find the promising region before spending optimizer budget. | Used for REGION FINDING only - the selected config records selection_policy 'optimizer_only_after_grid_region_finding' and 'supersedes: previous grid-selected latent8_v002 config', so an earlier grid-selected config existed and was replaced by the optimizer-selected one. | Grid output directory no longer exists on disk (cleanup Wave 1). | The grid-selected config was superseded by the optimizer-selected one. | Bounded the optimizer search; explicitly NOT the selection step. | high |
| **Optimizer search v002 - SOURCE** | `latent8_ngrc_optimizer_search_v002.json (+ ..._tree_search_v002.json)` | optimizer_search | optimizers [gp, dummy]; continuous NGRC latent8 space | validation relative_l2_horizon_time (maximize) | Select NGRC latent8 by continuous validation-only search after grid region finding. | The DUMMY (random) optimizer produced the winner: selection_source 'v002_optimizer_dummy'. Search output ngrc_latent8_optimizer_search_v002 survives on disk and backs the 3 optimizer figures. | That random search won suggests the latent8 NGRC objective is not strongly structured - a broad flat landscape. This is an inference from the selection_source field, not an explicit statement. | n/a - produced the final config. | IS the source of the final selected config. | high |
| **Optimizer search v003 (not adopted)** | `latent8_ngrc_optimizer_search_v003.json` | optimizer_search | v003 space | validation relative_l2_horizon_time | A v003 latent8 optimizer-search CONFIG exists. | NO v003 latent8 selected config was ever produced, and no v003 latent8 search output exists on disk. The package uses selected v002. | Whether v003 ran at all, and why it was not adopted, is unknown from available evidence. | n/a - the final result stayed at v002. | Not connected to the final result; recorded so the version gap is not mistaken for an omission. | low |
| **Selected v002** | `latent8_ngrc_selected_v002.json` | selected_evaluation | n/a (locked config) | validation horizon for selection; test evaluated once after locking | Lock the optimizer-selected latent8 config and report test once. | valid 24.2 / test 9.8. | NGRC latent8 is deterministic (no RNG) - no seed distribution exists. Its quality is bounded by the latent-8 representation, not only by the forecaster. | n/a - final. | IS the final selected config. | high |
| **Package reproduction (Stage 4)** | `final_thesis_package_v001/configs/02_forecasting/ngrc_latent8.json` | package_reproduction | n/a (single locked config) | validation/test relative_l2_horizon_time vs canonical | Re-execute the locked selected config inside the package against package-local data. | Reproduced canonical to the decimal on the conda torch interpreter forced to CPU (CUDA_VISIBLE_DEVICES=''). Registry unchanged. | none identified at this step | n/a - accepted final form. | IS the package result: results/02_forecasting/ngrc/selected/ngrc_latent8_selected_v002 | high |
| **Figures (Stage 6)** | `final_thesis_package_v001/plots/02_forecasting/per_model/ngrc_latent8/ (4 PNGs)` | plotting | n/a | n/a | Generate the per-model forecasting figures from the accepted package results. | 4 PNGs: {validation,test}_{relative_l2_horizon,spatiotemporal_comparison}. Lyapunov time axes (use_lyapunov=true, lambda_max=0.043). | Figure time axes are in Lyapunov units, so horizon markers do NOT equal the physical-time tables. | n/a - terminal step. | Supplies this model's thesis-facing figures. | high |

## 4. Final selected config/result

**Package config:** `configs/02_forecasting/ngrc_latent8.json` (`run_id: ngrc_latent8_selected_v002`)
**Canonical source:** `latent8_ngrc_selected_v002.json`
**Result:** `results/02_forecasting/ngrc/selected/ngrc_latent8_selected_v002/`

Documented selection metadata: `selection_policy: optimizer_only_after_grid_region_finding`,
`selection_source: v002_optimizer_dummy`, `supersedes: previous grid-selected latent8_v002 config`.
Latent predictions are **decoded back to 64D physical space before evaluation** (project rule: all final
comparisons happen in physical space).

## 5. Final metrics

> **Units.** Validation and test horizons are **physical simulation time**. The per-model forecasting
> figures plot **Lyapunov time** (`t·λ_max`, λ_max = 0.043), so a horizon marker on those axes equals
> `physical × 0.043` and will **not** match the numbers below. **Quote this table, never the axes.**

| metric | value |
|---|---|
| **validation horizon** | **24.2** (physical simulation time) |
| **test horizon** | **9.8** (physical simulation time) |
| seed | n/a — **deterministic**, no RNG |

## 6. Supporting plots

**Main result plots** (`plots/02_forecasting/per_model/ngrc_latent8/`)
- `test_relative_l2_horizon.png`, `test_spatiotemporal_comparison.png`.

**Selection/provenance plots** (`plots/optimizer_search/ngrc/ngrc_latent8/`)
- `optimizer_convergence.png` — genuine selection evidence (deterministic model).
- `optimizer_sampling_matrix.png`, `optimizer_surrogate_landscape.png`.

**Appendix/diagnostic plots**
- `validation_relative_l2_horizon.png`, `validation_spatiotemporal_comparison.png`.

*Note: the optimiser figures can be regenerated from the compact public trial-history dataset under
`forecasting/data/optimizer_search/`; identical PNG checksums across environments are not guaranteed.*

## 7. Thesis-facing statement

The same deterministic NGRC method was applied in the 8-dimensional autoencoder latent space, with latent
predictions decoded back to the physical field before evaluation. Selection followed a two-stage validation-only
protocol recorded in the config metadata: a coarse grid was used for **region finding** only, and the final
configuration was then chosen by a continuous optimizer search (`optimizer_only_after_grid_region_finding`),
superseding an earlier grid-selected configuration. It reaches a validation horizon of 24.2 and a held-out test
horizon of 9.8 in physical simulation time. The comparison with the full-field NGRC result is the point: holding the
method fixed and changing only the space, the test horizon falls from 324.0 to 9.8. Latent-space forecasting here is
bounded by the **representation**, not by the forecaster.

## 8. Notes and caveats

- **Do not** attribute the gap to NGRC. The method is identical to `ngrc_full64`; the **representation** is the
 binding constraint.
- **Do not** describe it as seed-selected. It is **deterministic** — no seed distribution exists, and no
 seed-robustness boxplot is definable.
- **Do** note that the winning configuration came from the **`dummy` (random) optimizer**
 (`selection_source: v002_optimizer_dummy`). *Available evidence suggests* this indicates a broad, weakly structured
 objective landscape, but that reading is an inference from the selection-source field, not a documented statement.
- A `latent8_ngrc_optimizer_search_v003` **config exists but no v003 selected config was ever produced** and no v003
 output survives. Whether it ran, and why it was not adopted, is **unknown from available evidence** — the version
 gap is recorded so it is not mistaken for an omission.
- Selection-process figure question **deferred**.
