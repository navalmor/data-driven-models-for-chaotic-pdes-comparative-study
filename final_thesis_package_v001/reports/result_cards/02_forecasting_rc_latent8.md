# Result Card — `rc_latent8`

## 1. Result identity

**RC (Reservoir Computing) forecasting in the 8-dimensional AE-SINDy latent space.**
It shares the full64 model's seed-luck problem and additionally inherits the latent-8 representation bound.

**Thesis role: APPENDIX.** RC (Reservoir Computing) is **stochastic**: its final model was **not** taken at the optimizer's argmax but
chosen by a **validation-only 15-seed rerank** of the top-10 candidates. 

## 2. Inputs and dependencies

| item | path |
|---|---|
| package config | [`configs/02_forecasting/rc_latent8.json`](../../configs/02_forecasting/rc_latent8.json) |
| canonical source config | `latent8_rc_selected_v003_seedrobust.json` |
| result | `results/02_forecasting/rc/selected/rc_latent8_selected_v003_seedrobust/` |
| rerank provenance (staged) | `results/02_forecasting/provenance/seed_robust/*.csv` |
| search source (survives) | `rc_latent8_optimizer_search_v002` |
| rerank archive | `final_stochastic_seed_robust_repair_v001` |

**Depends on:** the package's **own** latent export
`results/01_ae_sindy/latent8_trial014_representation/latent_data/`, which depends on the frozen simulation. Latent
predictions are **decoded back to 64D physical space before evaluation**.

## 3. Development and selection timeline

*Reconstructed from configs, config metadata, reports and on-disk artifacts. Where motivation is explicitly
documented it is cited; where it is inferred, the wording says so; where evidence is absent it says
"unknown from available evidence."*

| Step | Artifact/config/script | Step type | Candidate/search space | Metric or check used | Aim | Outcome / what was learned | Limitation or issue found | Why next step was needed | Connection to final result | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| **Smoke test** | `latent8_rc_smoke_v001.json` | smoke | minimal execution, not a search | runner completes; data path resolves; outputs written | Check the runner, config schema and data path execute end-to-end before spending compute. | Pipeline executes for this model/data_mode. Available evidence suggests no scientific claim was drawn from this stage. | A smoke run says nothing about model quality. | A parameter feasibility probe was needed before any search. | Infrastructure only; no bearing on the final numbers. | medium |
| **Probes** | `latent8_rc_probe_{reservoir1000_conservative,reservoir1000_regularized,slow_leak_washout200,small_input,small_radius,strong_ridge}_v001.json (6 configs)` | probe | the same 6 contrasting reservoir regimes as full64, in latent-8 space | validation relative_l2_horizon_time | Probe feasible regions before committing search budget. Configuration differences indicate each probe isolates one structural regime. | Available evidence suggests the probes bounded the search space; per-probe outcomes are not retained on disk. | Probes cannot resolve continuous interactions. | A grid then an optimizer search were needed. | Fixed the structural choices the later searches held constant. | medium |
| **Grid search v001** | `latent8_rc_grid_search_v001.json (+ latent8_rc_legacy_focus_search_v001.json)` | grid_search | coarse grid (ranges in config) | validation relative_l2_horizon_time | Coarse region finding before the optimizer. | Region-finding only. Grid output directories no longer exist on disk (cleanup Wave 1); evidenced by config. | The exact best grid point is unknown from available evidence. | An optimizer search over continuous parameters was the next step. | Bounded the optimizer search; not selection evidence. | medium |
| **Optimizer search v001 -> selected v001** | `rc_latent8_optimizer_search_v001 -> selected v001` | optimizer_search | continuous space; multiple optimizer families (dummy/forest/gbrt/gp) | validation relative_l2_horizon_time (maximize), single seed per trial | First continuous validation-only selection. | Produced a v001 selected config. SUPERSEDED. | Every optimizer trial trains with a DIFFERENT seed (optimizer_search_*.py assigns a different seed to every trial), so the search maximizes over (hyperparameters x seed) JOINTLY - hyperparameter quality and seed luck are inseparable. | A further search iteration was run before the defect was diagnosed. | Superseded; part of the chain that produced the published-but-inflated numbers. | medium |
| **Optimizer search v002/v003 -> published selected** | `rc_latent8_optimizer_search_v002 -> selected v002 (valid 42.4)` | optimizer_search | refined continuous space | validation relative_l2_horizon_time (maximize), single seed per trial | Refine the promising region and lock a selected config. | Produced the PUBLISHED config: valid_h 42.4 / test_h 14.7. Search directory rc_latent8_optimizer_search_v002 survives on disk and backs this model's 3 optimizer figures. | The published valid_h 42.4 lies ABOVE THE ENTIRE RANGE of fresh seeds - its seed was the winning trial's seed, i.e. selected for being lucky. This is winner's curse, not a better model. | A validation-only seed-robust rerank was required before any number could be defended. | SUPERSEDED by v003_seedrobust. Its 42.4 is the dashed reference line in the seed-robustness figure. | high |
| **Seed-robust rerank (15 seeds x 10 candidates)** | `final_stochastic_seed_robust_repair_v001 (750 runs total across 5 models)` | seed_rerank | top-10 candidates from rc_latent8_optimizer_search_v002 x 15 fresh seeds (9403101-9403115) = 150 validation runs for this model | median valid_relative_l2_horizon_time over 15 seeds, then mean, then min, then median valid_relative_l2_mean, then smaller candidate rank; validation only | Re-rank the top candidates by a seed-ROBUST statistic so hyperparameter quality is separated from seed luck. | Winner = candidate c02 (forest-sourced): median 20.0, min 13.8, max 26.1 over 15 seeds. Not one published candidate survived reranking across the five models. | The luckiest seed would have reported 26.1 - reporting it would have been noise. The spread is large relative to differences between candidates. | The winner then needed a representative (median) seed locked, and one authorized test evaluation. | Selects candidate c02 - IS the final selection decision for this model. | high |
| **Selected v003_seedrobust** | `latent8_rc_selected_v003_seedrobust.json` | selected_evaluation | n/a (locked config) | validation median for selection; test evaluated exactly once after locking (test_used_for_selection: false) | Lock the median-representative seed and report test once. | random_seed 9403102 - the rerank seed whose validation horizon is CLOSEST TO THE 15-SEED MEDIAN (abs difference from median 0.0). Final: valid 20.0 / test 16.6. | Deliberately NOT the best-performing seed (luckiest = 26.1). The locked model must be typical of the seed distribution, not a lucky tail draw. | n/a - final. | IS the final selected config. | high |
| **Seed-robustness figure (Stage 7B)** | `final_thesis_package_v001/plots/comparison/seed_robustness/rc_latent8/seed_robust_candidate_boxplots.png` | plotting | 10 candidates x 15 seeds | n/a (rendering only) | Make the selection correction legible per model. | Boxplot of 10 candidates x 15 seeds; green = winner c02, dashed line = published 42.4. | y-axis is validation horizon in PHYSICAL simulation time - unlike the per-model forecasting figures, which are in Lyapunov time. | n/a - terminal step. | The appendix evidence for this model's selection. | high |
| **Package reproduction (Stage 4)** | `final_thesis_package_v001/configs/02_forecasting/rc_latent8.json` | package_reproduction | n/a (single locked config) | validation/test relative_l2_horizon_time vs canonical | Re-execute the locked selected config inside the package against package-local data. | Reproduced canonical to the decimal on the conda torch interpreter forced to CPU (CUDA_VISIBLE_DEVICES=''). Registry unchanged. | none identified at this step | n/a - accepted final form. | IS the package result: results/02_forecasting/rc/selected/rc_latent8_selected_v003_seedrobust | high |
| **Figures (Stage 6)** | `final_thesis_package_v001/plots/02_forecasting/per_model/rc_latent8/ (4 PNGs)` | plotting | n/a | n/a | Generate the per-model forecasting figures from the accepted package results. | 4 PNGs: {validation,test}_{relative_l2_horizon,spatiotemporal_comparison}. Lyapunov time axes (use_lyapunov=true, lambda_max=0.043). | Figure time axes are in Lyapunov units, so horizon markers do NOT equal the physical-time tables. | n/a - terminal step. | Supplies this model's thesis-facing figures. | high |

## 4. Final selected config/result

**Package config:** `configs/02_forecasting/rc_latent8.json`
**Canonical source:** `latent8_rc_selected_v003_seedrobust.json`
**Result:** `results/02_forecasting/rc/selected/rc_latent8_selected_v003_seedrobust/`

**Selected candidate `c02`** (forest-sourced, from `rc_latent8_optimizer_search_v002`), **locked seed `9403102`** — the rerank seed whose
validation horizon is **closest to the 15-seed median** (`abs_difference_from_median: 0.0`), deliberately **not** the
luckiest seed (26.1). `test_used_for_selection: false`. Supersedes the published config (valid 42.4).

## 5. Final metrics

> **Units.** Validation and test horizons are **physical simulation time**. The per-model forecasting
> figures plot **Lyapunov time** (`t·λ_max`, λ_max = 0.043), so a horizon marker on those axes equals
> `physical × 0.043` and will **not** match the numbers below. **Quote this table, never the axes.**

| metric | value |
|---|---|
| **validation horizon** (15-seed **median**) | **20.0** (physical simulation time) |
| **test horizon** | **16.6** (physical simulation time) |
| locked seed | **9403102** (median-representative) |
| 15-seed validation range for the winner | min **13.8** — max **26.1** |
| superseded published value | 42.4 validation / 14.7 test |

The reported 20.0 is a **median across 15 seeds**, not a single lucky run.

## 6. Supporting plots

**Main result plots** (`plots/02_forecasting/per_model/rc_latent8/`)
- `test_relative_l2_horizon.png`, `test_spatiotemporal_comparison.png`.

**Selection/provenance plots** — **the primary selection evidence**
- `plots/comparison/seed_robustness/rc_latent8/seed_robust_candidate_boxplots.png` — **10 candidates × 15 seeds**
 (150 points). Green = the locked winner **c02**; orange = the originally published candidate; dashed line = its
 published value (42.4). **y-axis is validation horizon in PHYSICAL simulation time** — unlike the per-model
 forecasting figures above, which are in Lyapunov time.

**Appendix/diagnostic plots**
- `validation_relative_l2_horizon.png`, `validation_spatiotemporal_comparison.png` — selection split.
- `plots/optimizer_search/rc/rc_latent8/optimizer_*.png` (3) — **DIAGNOSTIC ONLY**: single-seed, search-time scores from
 a search whose argmax was **not** taken. The optimiser figures can be regenerated from the compact public
 trial-history dataset under `forecasting/data/optimizer_search/`; identical PNG checksums across environments
 are not guaranteed.

## 7. Thesis-facing statement

RC (Reservoir Computing) is a stochastic model, so its final configuration was not taken at the optimizer's argmax. The
validation-only optimizer search produced a published candidate reporting a validation horizon of 42.4, but that
value lies **above the entire range of fresh seeds**: because the search assigns a different random seed to every
trial, it ranks hyperparameters and seed luck jointly, and it had crowned a lucky draw rather than a better model. To
separate the two, the top ten candidates were re-run across **15 fresh seeds** and re-ranked by their **median**
validation horizon — a validation-only procedure in which no test metric was ever consulted. The winner is candidate
**c02** (which the original search had ranked 2nd), with a median validation horizon of **20.0** (range 13.8–26.1). The locked model uses the seed whose
validation horizon is closest to that median rather than the best seed (26.1), so the reported model is *typical* of
its seed distribution rather than a tail draw. On the held-out test split it reaches **16.6**. Note that its test horizon (16.6) is *higher* than the superseded config's test horizon (14.7) even though the
validation number fell sharply — the repair corrected an inflated validation estimate without costing test skill.

## 8. Notes and caveats

- **Do not** quote the superseded **42.4**. It was a lucky seed, not a better model.
- **Do not** describe the selection as "best seed" or "best search score". Selection used the **median validation
 horizon across 15 seeds**, and the locked seed is the **median-representative** one, not the luckiest (26.1).
- **Do not** read the optimizer-search figures as performance. They show single-seed search-time scores whose argmax
 was deliberately **not** taken; the **seed-robustness boxplot** is the real selection evidence.
- **Do** report 20.0 with awareness of its spread (13.8–26.1 across 15 seeds).
- **Do not** compare the seed-robustness y-axis (physical time) with the per-model figure axes (Lyapunov time).
- **Do not** attribute its ceiling to RC alone. Like every latent-8 model it is bounded by the **representation**;
 compare against `rc_full64` before concluding anything about the method.
