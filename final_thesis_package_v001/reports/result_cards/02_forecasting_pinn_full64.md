# Result Card — `pinn_full64`

## 1. Result identity

**PINN (Physics-Informed Neural Network) forecasting the full 64-dimensional physical KSE field**, finalized through a
**validation-only, seed-robust selection with a genuinely active physics term**.

**Thesis role: APPENDIX.** This is the most heavily audited result in the package. It supersedes a published
configuration whose physics weight was **zero**, and it must be reported with both its **seed spread** and its
**validation-to-test drop** rather than as a point estimate.

## 2. Inputs and dependencies

| item | path |
|---|---|
| package config | [`configs/02_forecasting/pinn_full64.json`](../../configs/02_forecasting/pinn_full64.json) |
| canonical source config | `full64_pinn_selected_v003_physics_active.json` |
| result | `results/02_forecasting/pinn/selected/pinn_full64_selected_v003_physics_active/` |
| search source | `pinn_full64_v003_physics_active_p3_{dummy,forest,gbrt,gp}_attempt01` |
| public search data | `forecasting/data/optimizer_search/pinn_full64/{dummy,forest,gbrt,gp}/optimizer_results.csv` |
| seed-robustness provenance | `results/02_forecasting/provenance/seed_robust/` |
| ablation provenance | `results/02_forecasting/provenance/pinn_full64_physics_ablation/` |

**Depends on:** `results/00_simulation/u_series.npy` **only** — no autoencoder involved.
**Runtime profile:** `scripts/repro_env.sh` (single-threaded, `PYTHONHASHSEED=0`). This is part of the result, not an
environment detail.

## 3. Development and selection timeline

*Reconstructed from configs, config metadata, gate reports and on-disk artifacts. Steps 1–5 are the pre-v002 history
and are retained unchanged; steps 6–8 are the superseded v002 chain; steps 10–16 are the physics-active repair.*

| Step | Artifact/config/script | Step type | Aim | Outcome / what was learned | Limitation or issue found | Connection to final result |
|---|---|---|---|---|---|---|
| **1. Smoke test** | `full64_pinn_smoke_v001.json` | smoke | Check runner, schema and data path execute end-to-end. | Pipeline executes. No scientific claim drawn. | A smoke run says nothing about model quality. | Infrastructure only. |
| **2. Probes** | `full64_pinn_{sanity_probe_v001,grid_probe_v001,training_probe_v002}.json` | probe | Probe training behaviour before committing budget. | Staged escalation sanity → grid → training. | Probes cannot resolve hyperparameter interactions. | Fixed structural choices later searches held constant. |
| **3. Grid search v001** | `pinn_full64_grid_search_v001` | grid_search | Coarse region finding. | `grid_best_validation_horizon` 22.8. | Grid resolution insufficient to separate candidates. | Bounded the optimizer search. |
| **4. Optimizer search v001 → selected v001** | `full64_pinn_optimizer_search_v001.json` | optimizer_search | Select by a four-family search. | gp trial 20 won at valid_h 27.7. | Reproducible only at an unrecorded `OMP_NUM_THREADS=16`; search confounded seed with hyperparameters. | SUPERSEDED. |
| **5. Stability repair audit** | (internal audit, not shipped) | documentation | Diagnose apparent irreproducibility. | The model was **never irreproducible**; the real defects were thread-dependence and a selection defect. | Thread-dependence plus seed confounding. | Set the v002 repair design. |
| **6. Deterministic optimizer search v002** | `full64_pinn_optimizer_search_v002_deterministic_*` | optimizer_search | Re-run the search with thread noise removed. | 320 deterministic trials; zero reach 27.7. | **Every trial carried `lambda_phy = 0.0`**, so it could not select a physics-active configuration. | SUPERSEDED by the P3 physics-active search. |
| **7. Seed-robust rerank v002 (staged)** | `seed_robust_rerank_v002{,_extension10}` | seed_rerank | Separate hyperparameter quality from seed luck. | Winner c05 = dummy trial 49, median 14.7 over 15 seeds. | Re-ranked zero-weight candidates; staged 4-candidate shape, not a uniform grid. | SUPERSEDED by the P4 uniform 10 × 15 grid. |
| **8. Selected v002_deterministic** | `full64_pinn_selected_v002_deterministic.json` | selected_evaluation | Lock c05 and report test once. | Seed 9201101; valid 14.7 / test 30.0. | **DECISIVE DEFECT found after publication: `lambda_phy = 0.0`.** The physics residual was never computed; the case was physics-informed in name only. Its numbers measure a plain residual network. | SUPERSEDED by selected v003. |
| **9. Seed-robustness figure gap (Q-36) — RESOLVED** | `plots/comparison/seed_robustness/pinn_full64/` | documentation | Decide whether this model gets a boxplot. | Originally none: the v002 rerank had a staged 4-candidate shape, not the uniform 10 × 15 grid the plot-ready table holds. **Resolved at P6** once the P4 grid replaced it. | The original absence was a shape mismatch, not a missing rerank. Recorded so the historical gap is not erased. | The model now carries the same primary selection-evidence figure as the other five stochastic models. |
| **10. P0–P1 physics-inactivity discovery** | Gate-1 evidence | documentation | Establish whether the published model computed a physics residual. | Confirmed `lambda_phy = 0.0` in the selected config and in all 320 v002 trials. The physics code was correct but multiplied by zero. | Every downstream v002 number describes a zero-weight model. | Defines why v003 exists. No v002 number was reused as evidence for v003. |
| **11. P2 physics activation and gradient audit** | Gate-2 evidence | verification | Prove a positive weight produces a real, finite gradient. | Raw physics-gradient norm 3119.1953125; weighted 0.0003851171811021172; **0 of 854080** elements nonfinite. Gradient audit **PASS**. | The weighted contribution is small: genuinely active but modest, not dominant. | Supplies the activity rule every later candidate had to satisfy. |
| **12. P3 physics-active optimizer search** | `forecasting/data/optimizer_search/pinn_full64/` | optimizer_search | Search hyperparameters with physics genuinely active. | 320 completed trials across four families; **201** reach the ≥ 1 % activity threshold. Search-time single-seed optimum: **C1 = GBRT trial 22 at VPH 31.2**. | A single-seed search score confounds hyperparameters with seed luck. | Supplies the 10-candidate pool. **The search-time optimum C1 was NOT taken.** |
| **13. P4 seed robustness, 10 × 15** | `provenance/seed_robust/seed_robust_candidate_distributions.csv` | seed_rerank | Separate hyperparameter quality from seed luck on a uniform grid. | **150 runs, 0 failures.** Winner **C5 = dummy trial 47**: median 19.8, mean 17.29, min 8.5, max 23.1, std 4.63; 15/15 activity-eligible. P4 ranking has Spearman **−0.224** against the P3 single-seed ranking. | Within-candidate seed spread is wide (8.5–23.1 for the winner). The median is robust, not tight. | **IS the final selection decision**, and resolves Q-36. |
| **14. P4A paired zero-weight ablation** | `provenance/pinn_full64_physics_ablation/` | ablation | Measure what the physics term contributes, holding all else fixed. | **VALIDATION ONLY.** Active median 19.8 vs zero-weight 14.8; median paired difference **+2.1**, mean +2.58; active better on **11/15**, zero-weight on **4/15**, 0 ties. | Single configuration; hyperparameters were tuned with physics active, which favours the active arm; zero-weight arm has marginally better post-threshold boundedness. Not a significance test. **No active-versus-zero comparison exists on test.** | The only controlled active-versus-zero evidence in the thesis, and validation-only by design. |
| **15. P5 lock, single test evaluation, reproduction** | `full64_pinn_selected_v003_physics_active.json` | selected_evaluation | Lock C5 at its median-representative seed, evaluate test once, prove reproducibility. | Seed **9301114** locked **before test access**. Final: **valid 19.8 / test 7.3**. Reproduction verdict **REPRODUCED_BITWISE** (23/23 scalars, 10/10 artifacts). | Test horizon far below validation; the rollout diverges strongly after threshold crossing. Reported as found; **did not trigger retuning, reseeding or alternative-candidate evaluation**. | IS the final selected result. |
| **16. P6 public package integration** | `results/02_forecasting/pinn/selected/pinn_full64_selected_v003_physics_active/` | package_integration | Publish the locked result and retire superseded surfaces. | The 14 artifacts were **transferred byte-for-byte** from the locked P5 output (11 identical; 3 path-bearing JSONs rewritten to package-relative paths with no scientific field changed). | Unlike the v002 entry, the package result was **not re-executed inside the package** — no model was run during P6. The config's literal scientific hash differs from the private one solely because `data.data_path` is rewritten; the path-invariant hash matches and the dataset bytes are identical. | Supplies the published form of the final result. |

## 4. Final selected config/result

**Package config:** `configs/02_forecasting/pinn_full64.json` (`run_id: pinn_full64_selected_v003_physics_active`)
**Canonical source:** `full64_pinn_selected_v003_physics_active.json`
**Result:** `results/02_forecasting/pinn/selected/pinn_full64_selected_v003_physics_active/`

**Selected candidate `C5`** = `dummy` **trial 47** of the v003 physics-active optimizer search
(`hidden_dim 512, depth 4, lr 3.3266127695166006e-04, multistep_horizon 3, beta_multistep 0.7521100511934925,`
`lambda_phy 1.2346683760352606e-07, weight_decay 1e-06, activation tanh, residual raw CNAB2`).
**Locked seed `9301114`** — the C5 rerank seed **closest to the 15-seed median** (`abs_difference_from_median: 0.0`),
deliberately **not** the best C5 seed (23.1, seed 9301112). **The seed was locked before any test access**;
`test_used_for_selection: false`.
**Supersedes** `full64_pinn_selected_v002_deterministic.json` (valid 14.7 / test 30.0, `lambda_phy = 0.0`).

## 5. Final metrics

> **Units.** Validation and test horizons are **physical simulation time**. The per-model forecasting
> figures plot **Lyapunov time** (`t·λ_max`, λ_max = 0.043), so a horizon marker on those axes equals
> `physical × 0.043` and will **not** match the numbers below. **Quote this table, never the axes.**

| metric | value |
|---|---|
| **validation horizon** (15-seed **median** of C5) | **19.8** (physical simulation time) |
| **test horizon** | **7.3** (physical simulation time) |
| locked seed | **9301114** (median-representative, locked pre-test) |
| C5 15-seed validation distribution | median 19.8, mean 17.29, min 8.5, max 23.1, **std 4.63** |
| test rollout | 73 steps; first unsafe index 73; mean rel-L2 32.69; final rel-L2 110.95; finite fraction 0.120 |
| physics activity (final 20 %) | **0.0205** ratio; gradient audit **PASS**; 0/854080 nonfinite |
| reproduction | **REPRODUCED_BITWISE** — 23/23 scalars exact, 10/10 artifacts byte-identical |
| paired validation-only ablation | active 19.8 vs zero-weight 14.8; median paired difference **+2.1**; active better 11/15 |

## 6. Supporting plots

**Main result plots** (`plots/02_forecasting/per_model/pinn_full64/`)
- `test_relative_l2_horizon.png`, `test_spatiotemporal_comparison.png`.

**Selection/provenance plots — PRIMARY selection evidence**
- `plots/comparison/seed_robustness/pinn_full64/seed_robust_candidate_boxplots.png` — the 10 candidates × 15 seeds
  validation distributions, physical-time y-axis, winner **C5 at search rank 5** in green. This model has **no
  published-candidate marker**: the previously published v002 configuration came from a different (zero-weight)
  search and is **not** among candidates C1–C10, so marking one of them "published" would be false.
  **Caption:** *C5 was selected by the highest median validation horizon across 15 seeds. The P3 single-seed
  search-time optimum (C1, search rank 1, 31.2) was not robust under seed repetition — its 15-seed median is 12.4.*
- `plots/comparison/pinn_full64_physics_ablation/paired_active_vs_zero_weight.png` — the paired ablation.
  **Caption:** *Paired validation-only ablation of the physics term for the selected PINN-64 configuration (C5).
  Each line joins one of 15 seeds evaluated with the physics weight active (λ = 1.2346683760352606e-07) and with it
  set to zero; no other setting differed. The active arm has the higher median valid-prediction horizon (19.8 versus
  14.8 time units, median paired difference +2.1) and is better on 11 of 15 seeds, while the zero-weight arm is
  better on 4 and shows marginally better post-threshold boundedness. This is a descriptive paired comparison for
  one selected configuration whose hyperparameters were themselves tuned with the physics term active, which favours
  the active arm; it is not a significance test and does not support a general claim about physics-informed
  training. No active-versus-zero comparison exists on the held-out test split.*

**Appendix/diagnostic plots**
- `validation_relative_l2_horizon.png`, `validation_spatiotemporal_comparison.png`.
- `plots/optimizer_search/pinn/pinn_full64/optimizer_*.png` (3) — **DIAGNOSTIC ONLY**; single-seed search-time scores
  whose argmax was **not** taken. The red star marks the search-time metric optimum (GBRT trial 22, 31.2), which is
  **not** the published configuration. These figures regenerate from the public trial-history dataset under
  `forecasting/data/optimizer_search/`; identical PNG checksums across environments are not guaranteed.

## 7. Thesis-facing statement

The published PINN full64 result carried `lambda_phy = 0.0`. The physics residual was never computed, so the case was
physics-informed in name only and its 14.7 / 30.0 numbers measure a plain residual network. The repair activated the
physics term and redid the entire selection: a four-family, 320-trial search with a positive physics weight, of which
201 trials reach the one-percent activity threshold; then a uniform re-ranking of the top ten candidates across
fifteen fresh seeds — 150 validation runs, zero failures — selecting by **median** validation horizon, validation
only. The winner is candidate **C5**, whose fifteen-seed median is **19.8**. The search's own best single draw,
**C1 at 31.2**, fell to a fifteen-seed median of 12.4: the single-seed ranking correlates with the seed-robust one at
Spearman **−0.224**, which is precisely why the repetition was run. A paired ablation on this exact configuration,
holding everything but the physics weight fixed, favours the active arm on validation (19.8 versus 14.8, median
paired difference +2.1, better on 11 of 15 seeds). The configuration and its median-representative seed **9301114**
were locked before any test access, and the single test evaluation returned a horizon of **7.3** — far below the
validation median, with the rollout diverging strongly after threshold crossing. That drop is reported as found: it
did not trigger retuning, reseeding or the evaluation of any alternative candidate, because doing so after test
access would destroy the validation-only guarantee that makes the number meaningful. The physics contribution is
real and verified but modest in loss terms, and the evidence that it helps is **validation-only and specific to this
one configuration**.

## 8. Notes and caveats

- **Do not** quote **27.7** or **30.0** as this model's result. 27.7 came from a thread-dependent, seed-confounded
  search; 30.0 belongs to the superseded zero-weight v002 configuration.
- **Do not** claim that **physics improved test performance**. No controlled active-versus-zero comparison exists on
  the test split, by design — the ablation is validation-only.
- **Do not** treat the **v003-versus-v002 difference as isolating physics**. The two differ in architecture,
  hyperparameters, seed and selection workflow; their test values are not comparable in that way.
- **Do not** claim statistical significance or global optimality. The ablation is descriptive: no p-value, no
  confidence interval. C5 is the pre-registered rule's answer over ten candidates, not a demonstrated optimum.
- **Do not** generalise to physics-informed learning as a class. This is one configuration, on one equation, whose
  hyperparameters were themselves tuned with the physics term active — an asymmetry that favours the active arm.
- **Do** report the **validation-to-test drop plainly** (19.8 → 7.3). It is the honest headline for this case.
- **Do** report the **seed spread** (min 8.5, max 23.1, std 4.63) alongside the median.
- **Do** note that **4 of 15 seeds favour zero weight**, two by margins comparable to the median gain, and that the
  zero-weight arm shows marginally better post-threshold boundedness (finite fraction 0.1074 versus 0.0946).
- **Do** state that the physics term is **genuinely active but modest**: final-20 % activity ratio 0.0205, weighted
  gradient norm 3.85e-04 against a raw norm of 3119.2.
- **Do not** state that this model was not seed-reranked — it was, on a uniform **10 × 15 = 150-run** grid, the same
  shape the other five stochastic models received. Q-36 is **resolved**.
- **Do** treat the single-threaded runtime profile as part of the result specification.
