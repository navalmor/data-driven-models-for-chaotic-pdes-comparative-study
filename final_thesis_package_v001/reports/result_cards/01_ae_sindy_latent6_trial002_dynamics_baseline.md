# Result Card — `latent6_trial002_dynamics_baseline`

## 1. Result identity

The **latent-6 discovered-dynamics baseline** (`trial_002` of Optuna Stage 2): the thesis's **interpretability
contribution** — 17 sparse quadratic terms describing the KSE's dominant latent couplings.

**Thesis role: MAIN** (the interpretable result). It is a *dynamics* model, selected by rollout horizon — a
different role and a different criterion from the latent-8 representation.

## 2. Inputs and dependencies

| item | path |
|---|---|
| config | [`configs/01_ae_sindy/latent6_trial002_dynamics_baseline_checkpoint_reanalysis.json`](../../configs/01_ae_sindy/latent6_trial002_dynamics_baseline_checkpoint_reanalysis.json) |
| result | `results/01_ae_sindy/latent6_trial002_dynamics_baseline/` |
| frozen checkpoint source | `optuna_stage2_trial_002` (`model.pt`) *(reference/provenance only)* |

**Depends on:** `results/00_simulation/u_series.npy` + the **frozen `model.pt`** (sha256 `30c3f2e1…8301d8294`,
byte-identical to source).
**Consumed by:** nothing. It is a standalone baseline; no forecasting model uses it.

## 3. Development and selection timeline

*Reconstructed from configs, config metadata, reports and on-disk artifacts. Where motivation is explicitly
documented it is cited; where it is inferred, the wording says so; where evidence is absent it says
"unknown from available evidence."*

| Step | Artifact/config/script | Step type | Candidate/search space | Metric or check used | Aim | Outcome / what was learned | Limitation or issue found | Why next step was needed | Connection to final result | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| **Manual grid sweeps v1 -> v2 -> v3** | `scripts/generate_ae_sindy_sweep_v{1,2,3}.py -> configs/ae_sindy/sweep_v{1,2,3}/ (24 / 13 / 72 configs)` | grid_search | v1: threshold {0.03..0.06} x sindy_z {0.4,0.6,0.8} x reg {3e-4,1e-3}; v2 re-centred on v1 winner, adds sindy_x {0.01,0.02}; v3: dense local refinement (72 runs) | analysis-window rollout horizon + reconstruction | Map the latent-6 sparsity/loss landscape systematically; each version re-centres on the previous winner. | v1 best sat at the grid EDGE (thr~0.06, z~0.8) -> v2 re-centred and added sindy_x -> v2_003 best -> v3 confirmed a SHALLOW, FLAT optimum. | v1's optimum was at the grid edge and sindy_x was unvaried; by v3 the objective was flat - manual grids waste runs on flat regions and cannot search continuously. | Flatness + edge effects motivated Bayesian (Optuna) optimisation. | Narrowed the latent-6 threshold/weight region that Optuna then refined into trial_002. | high |
| **Optuna Stage 1 (latent-6 broad)** | `scripts/optuna_ae_sindy_stage1.py; study optuna_stage1 (~28 trials, TPE)` | broad_search | coefficient_threshold [0.03,0.08], sindy_z [0.4,1.2], sindy_x [0.001,0.03] log, regularization log, learning_rate [5e-5,5e-4] log | SINDy rollout horizon (primary) + decoder metric, active terms, rollout success | Replace manual gridding with continuous Bayesian search over latent-6. | Identified a high-threshold, moderate-z sub-region as most promising. | Stage-1 ranges were broad; the best region deserved higher resolution. | A focused Stage 2 could exploit the best region. | Bounds the focused Stage 2 that produced the final baseline. | high |
| **Optuna Stage 2 (focused latent-6) - SOURCE** | `scripts/optuna_ae_sindy_stage2.py; study optuna_stage2 (25 trials); run optuna_stage2_trial_002` | focused_search | coefficient_threshold [0.060,0.085], sindy_z [0.35,0.75], sindy_x [0.0006,0.004] log, learning_rate [5e-5,1.8e-4] log | robust multi-window VALIDATION horizon (mean + worst case) + validation reconstruction + parsimony | Finalise the discovered-dynamics candidate in the best Stage-1 region. | trial_002: robust rank 1 by mean validation horizon (4.30, min 1.10) AND best validation reconstruction (0.0204) of the study; 17 active quadratic terms; 5/5 windows stable. | Even the best latent-6 dynamics have a short, window-variable horizon (intrinsic to chaotic KSE); one latent coordinate (z2) carries no above-threshold dynamics. | Library choice still had to be justified, and the horizon/reconstruction trade-off motivated separating the representation role. | IS the latent-6 dynamics baseline. | high |
| **Library ablation** | `library_ablation (lib_poly1_*, lib_poly2_sine_*, lib_poly3_*)` | grid_search | poly_order in {1,2,3} and a poly2+sine variant | multi-window horizon + reconstruction | Justify the SINDy library choice rather than assuming it. | poly_order=2 (no sine) is the right complexity: order 1 underfits, order 3 and sine add terms without improving robust horizon. | none - this closes the library question. | n/a - question closed. | Justifies the quadratic library of the final baseline. | high |
| **Package from-scratch retrain ATTEMPT - REJECTED** | `option A (attempted)` | package_reproduction | n/a (single locked config) | reproduction of the canonical checkpoint metrics | Try to rebuild latent6_trial002 from scratch inside the package, as done for latent8. | FAILED and was REJECTED: cross-environment CUDA nondeterminism means the checkpoint is not from-scratch bit-reproducible on today's hardware. | The retrain did not reproduce the canonical selected checkpoint; a mismatch forensic was required. | Option B (frozen checkpoint + package-local reanalysis) was adopted instead. | Explains WHY the final package result is a reanalysis rather than a retrain. Needed for honesty. | high |
| **Frozen checkpoint reanalysis (Stage 4A, Option B)** | `final_thesis_package_v001/configs/01_ae_sindy/latent6_trial002_dynamics_baseline_checkpoint_reanalysis.json` | checkpoint_reanalysis | n/a (frozen weights; no training) | reproduction of canonical horizon / decoder error / active terms | Reanalyse and visualise the FROZEN selected checkpoint package-locally, without retraining. | Reproduced canonical EXACTLY: test horizon 8.8, decoder_error 0.022894755005836487, active_terms 17. model.pt byte-identical to source. | retrained: false. The outcome IS the frozen selected checkpoint - NOT a from-scratch retrain. | n/a - this is the accepted final form. | IS the package's latent-6 dynamics baseline. | high |
| **Figures** | `final_thesis_package_v001/plots/01_ae_sindy/latent6_trial002_dynamics_baseline/ (6 PNGs)` | plotting | n/a | n/a | Produce and curate the run-native figures. | 6 figures; 4 on Lyapunov time axes. | Rendered from a frozen checkpoint, not a retrain. | n/a - terminal step. | Supplies the latent-6 baseline figures. | high |

## 4. Final selected config/result

**Config:** `configs/01_ae_sindy/latent6_trial002_dynamics_baseline_checkpoint_reanalysis.json`
**Result:** `results/01_ae_sindy/latent6_trial002_dynamics_baseline/`

**Status: 🔒 FROZEN CHECKPOINT + package-local reanalysis (Option B).** `retrained: false`. A full from-scratch
retrain (option A) was **attempted and rejected** — cross-environment CUDA nondeterminism means the checkpoint
is not from-scratch bit-reproducible on today's hardware. The frozen `model.pt` was copied verbatim and
**reanalysed + visualised package-locally** on the package simulation. **This is not a from-scratch retrain.**

Reproduce with `run_analysis.py --eval-split test` then `run_visualize.py` — **not** `run_train.py`.

## 5. Final metrics

| metric | value | units |
|---|---|---|
| **test SINDy horizon** | **8.8** | **physical simulation time** |
| **test decoder error** | **0.022894755005836487** | dimensionless |
| **active terms** | **17** (of 168 quadratic candidates) | count |
| validation: mean multi-window horizon | 4.30 (min 1.10, std 2.41), 5/5 windows stable | physical time |
| validation decoder metric | 0.0204 — best of the latent-6 study | dimensionless |
| multi-window test | mean 4.68 / min 2.10 / max 8.80 / std 2.87 (5/5) | physical time |

The package reanalysis reproduced the canonical values **exactly**.

## 6. Supporting plots

**Main result plots** (`plots/01_ae_sindy/latent6_trial002_dynamics_baseline/`)
- `rollout_sindy_comparison.png` — the end-to-end discovered-dynamics rollout (**Lyapunov** time axis).
- `reconstruction.png`, `rollout_encoder_comparison.png`, `latent_space_comparison.png` (**Lyapunov**).

**Selection/provenance plots** — none in-package (Stage-2 study is reference only).
**Appendix/diagnostic plots**
- `xi_heatmap.png` — the 17 active quadratic terms, made visible (no time axis). The natural companion figure.
- `training_history.png` — diagnostic only (no time axis).

## 7. Thesis-facing statement

The interpretable latent dynamics are taken from `trial_002` of a focused 25-trial Bayesian search at latent
dimension 6, selected under a **robust multi-window validation criterion** — the mean and worst-case SINDy rollout
horizon across five disjoint validation windows, together with validation reconstruction and parsimony. It is
uniquely both the **rank-1 model by mean validation horizon** and the **best-reconstructing** latent-6 candidate,
and it integrates stably on every validation and test window, where many latent-6 candidates collapse on at least
one. On the held-out test window it reaches a rollout horizon of 8.8 with a decoder error of 0.023, using 17 active
quadratic terms. In the final package this result is shipped as a **frozen selected checkpoint reanalysed
package-locally**, not as a from-scratch retrain: a retrain was attempted and rejected because the checkpoint is not
bit-reproducible across CUDA environments, and the reanalysis reproduces the canonical metrics exactly.

## 8. Notes and caveats

- **Do not** claim the latent-6 dynamics are a **complete latent KSE model**. One latent coordinate (z2) carries no
 above-threshold dynamics and only quadratic terms survive — it captures the dominant quadratic couplings.
- **Do not** present the package result as a **from-scratch retrain**. It is a frozen-checkpoint reanalysis
 (`retrained: false`); the retrain attempt failed and was rejected on evidence.
- **Do not** quote 8.8 as a stable single number. Report it with its spread — multi-window test mean 4.68 ± 2.87.
 The variability is intrinsic to chaotic KSE latent dynamics, not a modelling defect.
- **Do not** cite the superseded trial019 as a thesis-facing result; it is internal provenance only.
- **Do not** conflate this with the latent-8 representation: different latent dimension, different role, different
 selection criterion.
