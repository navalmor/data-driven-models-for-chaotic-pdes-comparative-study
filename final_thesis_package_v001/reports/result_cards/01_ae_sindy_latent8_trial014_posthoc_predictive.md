# Result Card — `latent8_trial014_posthoc_thr0025_predictive`

## 1. Result identity

The **latent-8 post-hoc predictive refit**: on the *same frozen* latent-8 representation, the SINDy coefficient
matrix is **re-solved** for prediction. It answers a distinct question — how much autonomous latent-prediction skill
the representation can support — and yields a **dense** (~81 % active), predictive model.

**Thesis role: APPENDIX.** It is a *predictive* refit, explicitly **not** a sparse or interpretable equation.

## 2. Inputs and dependencies

| item | path |
|---|---|
| config | [`configs/01_ae_sindy/latent8_trial014_posthoc_thr0025.json`](../../configs/01_ae_sindy/latent8_trial014_posthoc_thr0025.json) |
| result | `results/01_ae_sindy/latent8_trial014_posthoc_thr0025_predictive/` |
| config actually used (recorded) | `results/01_ae_sindy/latent8_trial014_posthoc_thr0025_predictive/posthoc_sindy_refit_config_used.json` |

**Depends on:** the **frozen** `latent8_trial014_representation` run
(`results/01_ae_sindy/latent8_trial014_representation/repro_latent8_trial014`). The autoencoder is **not**
retrained — only the coefficients change.
**Consumed by:** nothing. The forecasting models never integrate these coefficients.

## 3. Development and selection timeline

*Reconstructed from configs, config metadata, reports and on-disk artifacts. Where motivation is explicitly
documented it is cited; where it is inferred, the wording says so; where evidence is absent it says
"unknown from available evidence."*

| Step | Artifact/config/script | Step type | Candidate/search space | Metric or check used | Aim | Outcome / what was learned | Limitation or issue found | Why next step was needed | Connection to final result | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| **Frozen representation dependency** | `final_thesis_package_v001/results/01_ae_sindy/latent8_trial014_representation/repro_latent8_trial014` | latent_export | n/a | n/a | Fix the representation the refit is built on. | The SAME frozen latent-8 encoder/decoder is reused; the autoencoder is NOT retrained - only SINDy coefficients change. | The refit is bounded by the representation it sits on. | n/a - dependency step. | Defines what 'post-hoc' means here: coefficients re-solved on a frozen representation. | high |
| **Post-hoc threshold x ridge trade-off study** | `, .../posthoc_sindy_refit_strongridge_latent8/` | grid_search | trade-off: ridge_alpha {1e-3,1e-4,1e-5,1e-6} x threshold {0.025..0.065}; strong-ridge: ridge_alpha {1.0,0.1,0.01} x threshold {0.05..0.15}; 60 candidates total | robust multi-window VALIDATION rollout horizon (mean + worst case) | Ask whether the frozen representation can support stronger autonomous latent prediction when coefficients are refit for prediction rather than sparsity. | Threshold 0.025 is rank 1 by BOTH mean (11.5) and worst-case (5.30) validation horizon; ridge alpha is immaterial in 1e-3..1e-6. | Predictive horizon and sparsity trade off DIRECTLY: sparser refits lose most of the horizon, so the predictive winner is dense (292/360 ~ 81% active) and NOT interpretable. | The validation-selected threshold could then be reported once on test. | Selects threshold 0.025 / ridge 0.001 - the final post-hoc configuration. | high |
| **Package post-hoc refit (Stage 5)** | `final_thesis_package_v001/configs/01_ae_sindy/latent8_trial014_posthoc_thr0025.json` | posthoc_refit | single locked point: threshold 0.025, ridge_alpha 0.001, max_iter 10 | 5 test windows [25000,30000] reported once; test_used_for_selection=false | Reproduce the validation-selected post-hoc refit inside the package and report test once. | mean test horizon 10.860000000000001 (per-window 7.6 / 10.5 / 19.5 / 11.5 / 5.2); 292/360 active; refit_train_mse 0.0003019851840719146; mean_decoder_error 0.005778396315872669. | Dense (81% active) - a PREDICTIVE refit, not a sparse/interpretable equation. The forecasting models never integrate these coefficients. | n/a - accepted final form. | IS the package's post-hoc predictive result. | high |
| **Zero post-hoc figures (decision)** | `final_thesis_package_v001/plots/ (no posthoc directory)` | documentation | n/a | n/a | Decide whether post-hoc figures ship. | ZERO post-hoc figures ship. The package deliberately contains no post-hoc plot directory. | The post-hoc trio in the old tree remains excluded; adding it would reopen the Stage-5 zero-post-hoc decision. | Held deferred. | This result is quoted from tables only, not shown as a figure. | high |

## 4. Final selected config/result

**Config:** `configs/01_ae_sindy/latent8_trial014_posthoc_thr0025.json` — threshold **0.025**, `ridge_alpha`
**0.001**, `max_iter` 10.
**Result:** `results/01_ae_sindy/latent8_trial014_posthoc_thr0025_predictive/`

Coefficients fit by ridge-regularised STLSQ on the **train** split `[0,20000)`; threshold selected on **validation**;
the five test windows `[25000,30000]` reported **exactly once** (`test_used_for_selection: false`, H-01 safeguard).

## 5. Final metrics

| metric | value | units |
|---|---|---|
| **mean test horizon** | **10.860000000000001** | **physical simulation time** |
| per-window test horizons | **7.6 / 10.5 / 19.5 / 11.5 / 5.2** (min 5.2, max 19.5, std 4.85) | physical time |
| **active terms** | **292 / 360** (~**81 % dense**) | count |
| `refit_train_mse` | **0.0003019851840719146** | dimensionless |
| `mean_decoder_error` | **0.005778396315872669** | dimensionless |
| validation (selection) | threshold 0.025 rank **1** by both mean (**11.5**) and worst-case (**5.30**) horizon | physical time |

Ridge α is **immaterial** across 1e-3…1e-6; 5/5 solver success.

## 6. Supporting plots

**Main result plots** — **none.** Zero post-hoc figures ship in this package (Stage-5 decision). This result is
quoted from `tables/ae_sindy_final_metrics.csv` only.

**Selection/provenance plots** — none in-package.
**Appendix/diagnostic plots** — none. The post-hoc trio in the legacy tree remains **excluded/pending**; adding it
would reopen the Stage-5 zero-post-hoc decision.

## 7. Thesis-facing statement

To test how much autonomous predictive skill the latent-8 coordinates can support, the autoencoder was frozen and
the SINDy coefficients re-solved post-hoc on the training split by ridge-regularised STLSQ. The sparsity threshold
was chosen over a 60-candidate threshold × ridge grid using the robust multi-window **validation** horizon: threshold
0.025 ranks first by both mean (11.5) and worst-case (5.30) validation horizon, with the ridge coefficient immaterial
across four orders of magnitude. On the held-out test windows the refit reaches a mean valid horizon of **10.86**
(range 5.2–19.5). The result is deliberately reported as a **dense predictive** model rather than an interpretable
one: the winning threshold leaves **292 of 360 coefficients (81 %) active**, and sparser refits lose most of the
horizon. Predictive horizon and sparsity trade off directly, and this model sits at the predictive end.

## 8. Notes and caveats

- **Do not** call this model **sparse** or **interpretable**. It is **81 % dense**. That density is what buys the
 horizon.
- **Do not** claim the forecasting models use these coefficients. They do not, and never integrate them.
- **Do not** present it as an improvement of the latent-6 discovered dynamics. It answers a *different question*, on a
 *different latent dimension*, with a *different criterion*.
- **Do not** claim test data informed selection. The threshold was chosen on validation; test was read once.
- **Do** report 10.86 with its spread (5.2–19.5, std 4.85) — window variance is intrinsic to chaotic latent dynamics.
