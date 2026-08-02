# Result Card — `latent8_trial014_representation`

## 1. Result identity

The **latent-8 representation**: the frozen encoder/decoder whose encoded trajectory is the coordinate system in
which the four latent-8 forecasting models (NGRC, RC, PERC, PINN) operate. Selected from Optuna Stage 4A as
`trial_014`.

**Thesis role: MAIN.** It is a *representation*, selected by **reconstruction fidelity** — **not** a dynamics model
and **not** a claim about sparse SINDy quality.

## 2. Inputs and dependencies

| item | path |
|---|---|
| config | [`configs/01_ae_sindy/latent8_trial014_representation.json`](../../configs/01_ae_sindy/latent8_trial014_representation.json) |
| result (run) | `results/01_ae_sindy/latent8_trial014_representation/repro_latent8_trial014/` |
| exported latents | `results/01_ae_sindy/latent8_trial014_representation/latent_data/` |
| canonical source run | `optuna_stage4_latent8_trial_014` *(reference/provenance only)* |

**Depends on:** `results/00_simulation/u_series.npy` (frozen dataset).
**Consumed by:** `ngrc_latent8`, `rc_latent8`, `perc_latent8`, `pinn_latent8` — via `latent_data/`, whose
`source_model_info.json` resolves entirely to **package-local paths**.

## 3. Development and selection timeline

*Reconstructed from configs, config metadata, reports and on-disk artifacts. Where motivation is explicitly
documented it is cited; where it is inferred, the wording says so; where evidence is absent it says
"unknown from available evidence."*

| Step | Artifact/config/script | Step type | Candidate/search space | Metric or check used | Aim | Outcome / what was learned | Limitation or issue found | Why next step was needed | Connection to final result | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| **Initial clean AE-SINDy baseline** | `configs/ae_sindy/clean_norm_best_v1.json` | probe | single hand-tuned config; latent_dim=6, poly_order=2 | none yet - establishes reconstruction + rollout-horizon machinery | Establish a working config-driven AE-SINDy pipeline on the canonical split. | Pipeline, split and analysis metrics established. | A single config cannot say which sparsity/loss regime is best. | A systematic search was needed. | Defines the metrics all three AE-SINDy outcomes are measured with. | high |
| **HPC baseline + sparse variants** | `configs/ae_sindy/hpc_v100_{baseline,sparse_v1,sparse_v2_xavier,sparse_v3_balanced}.json` | probe | ~4 contrasting configs: latent 6 vs 8, poly_order 2 vs 3, constant vs xavier init, threshold 0.05 vs 0.2 | reconstruction + rollout horizon (qualitative) | Probe qualitatively different regimes and settle structural defaults. | The xavier-init, poly_order=2 regime was most balanced; structural defaults settled. | Only a handful of points sampled; loss-weight/threshold interactions unresolved. | A proper grid was needed. | Settled poly_order=2 + xavier init used by the latent-8 study too. | high |
| **Optuna Stage 4A (latent-8) - SOURCE** | `scripts/optuna_ae_sindy_stage4_latent8.py; study optuna_stage4_latent8 (24 retained trials)` | focused_search | latent_dim=8 frozen; coefficient_threshold [0.070,0.090], loss_weight_sindy_z [0.50,0.75], loss_weight_sindy_x [0.0008,0.003] log, learning_rate [6e-5,1.5e-4] log | relative VALIDATION reconstruction metric, with worst-case validation horizon as robustness tiebreak | Explore latent dim 8, because downstream forecasting benefits from a richer representation than the 6D dynamics model. | trial_014 selected: relative validation reconstruction ~0.0055, best tier of all 24 candidates; worst-case-horizon rank 3/24 (never collapses). | The best-reconstruction latent-8 model is NOT the longest-horizon dynamics model - the two objectives genuinely diverge at latent-8. | That divergence is exactly why representation and dynamics are selected by different criteria, and it motivated the post-hoc refit. | IS the latent-8 representation: the frozen encoder/decoder all latent8 forecasting operates in. | high |
| **Optuna Stage 4B (latent-8 wider) - breadth check** | `scripts/optuna_ae_sindy_stage4b_latent8.py; study optuna_stage4b_latent8 (25 trials)` | broad_search | coefficient_threshold [0.085,0.170], sindy_z [0.25,1.00], sindy_x [0.0002,0.006] log, learning_rate [4e-5,2e-4] log, threshold_frequency {10,25,50,100} | multi-window validation horizon + reconstruction | Confirm the Stage-4A representation is not beaten by a wider region. | The wider search did NOT displace Stage-4A on validation reconstruction; trial_014 stands. | Confirmed no better latent-8 REPRESENTATION was available in the wider region. | Left open whether the frozen representation could support stronger autonomous PREDICTION - which motivated the post-hoc refit. | A breadth/robustness check that reinforces trial_014; not itself a final outcome. | high |
| **Seed robustness + multi-window evaluation** | `, multiwindow_eval` | seed_rerank | leading trials re-run under multiple seeds; 5 disjoint 1000-step windows | stability of horizon/reconstruction across seeds and windows | Confirm the selected candidates are not single-seed or single-window artefacts. | Leading candidates stable across seeds; latent-8 dynamics more robust than latent-6. | A single chaotic rollout horizon is high-variance and near-uncorrelated between windows. | Established the robust multi-window criterion used for all final AE-SINDy selection. | Supports the robustness claim for the representation. | high |
| **Package reproduction (Stage 3)** | `final_thesis_package_v001/configs/01_ae_sindy/latent8_trial014_representation.json` | package_reproduction | n/a (single locked config) | reconstruction metrics vs locked reference; max abs difference | Re-execute the locked representation from scratch inside the package on the frozen simulation. | REPRODUCED EXACTLY (max abs difference 0.0): MSE mean 0.006592637859284878, MAE mean 0.062012095004320145, latent (30106,8). | Requires a GPU to retrain the autoencoder (device_used: cuda); latent8 FORECASTING itself is CPU-reproducible from the export. | The exported latents had to be produced so latent8 forecasting could close its dependency inside the package. | IS the package's accepted representation and the dependency of all four latent8 forecasting models. | high |
| **Latent export** | `final_thesis_package_v001/results/01_ae_sindy/latent8_trial014_representation/latent_data/` | latent_export | n/a (deterministic export) | latent shape + byte-identical reproducibility | Encode the full trajectory into frozen latent-8 coordinates and export train/valid/test arrays for forecasting. | z_series (30106,8) + z_train/z_valid/z_test at 20000/25000; source_model_info.json points only at package-local paths. | none identified | Latent8 forecasting configs could then read the package's OWN export, closing the chain inside the package. | The exact interface RC/NGRC/PERC/PINN latent8 consume. | high |
| **Figures** | `final_thesis_package_v001/plots/01_ae_sindy/latent8_trial014_representation/ (6 PNGs)` | plotting | n/a | n/a | Produce the run-native figures and curate them into plots/. | 6 figures; 4 carry Lyapunov time axes (use_lyapunov=true, lambda_max=0.043). | Copied into plots/ rather than regenerated there. | n/a - terminal step. | Supplies the thesis-facing AE-SINDy figures. | high |

## 4. Final selected config/result

**Config:** `configs/01_ae_sindy/latent8_trial014_representation.json`
**Result:** `results/01_ae_sindy/latent8_trial014_representation/` — `repro_latent8_trial014/` (model.pt, analysis
summaries, figures) + `latent_data/` (z_series, z_train/z_valid/z_test, u_reconstructed, reconstruction summary).

**Frozen learned representation (2026-07-17 policy lock): checksum/shape-validated by default, not retrained.**
The Stage 3 GPU run on its **original same-stack profile** (RTX 2080 Ti, torch 2.6.0, seed 42,
`deterministic=true`) reproduced the canonical result **exactly** (max abs difference **0.0**), and an
optional opt-in same-stack historical-profile retrain re-confirmed the checkpoint and latent exports
**bit-identically** (2026-07-17) — **provenance evidence only, not the default reproduction path and not a
cross-hardware guarantee** (a legitimate retrain on different hardware need not match bit-for-bit). The
latent export closes the forecasting dependency chain *inside* the package.

## 5. Final metrics

| metric | value | units |
|---|---|---|
| reconstruction **MSE mean** | **0.006592637859284878** | dimensionless (normalised field units) |
| reconstruction **MAE mean** | **0.062012095004320145** | dimensionless |
| latent array `z_series` | **(30106, 8)** | steps × latent dim |
| relative validation reconstruction | ≈ **0.0055** — best tier of all 24 latent-8 candidates | dimensionless |
| worst-case validation horizon rank | **3 / 24** (never collapses) | rank |

Splits in the export: train `[0,20000)`, valid `[20000,25000)`, test `[25000,30106)`.
Verified directly against `latent_data/reconstruction_summary.csv`.

## 6. Supporting plots

**Main result plots** (`plots/01_ae_sindy/latent8_trial014_representation/`)
- `reconstruction.png` — truth / reconstruction / error triptych (**Lyapunov** time axis).
- `rollout_sindy_comparison.png`, `rollout_encoder_comparison.png` — rollout triptychs (**Lyapunov**).
- `latent_space_comparison.png` — latent trajectory per dimension (**Lyapunov**).

**Selection/provenance plots** — none in-package; selection evidence is the Stage-4A study (reference only).
**Appendix/diagnostic plots**
- `xi_heatmap.png` — SINDy coefficient matrix Ξ (no time axis).
- `training_history.png` — validation loss + active terms (no time axis; diagnostic only).

## 7. Thesis-facing statement

The latent-8 autoencoder `trial_014` was selected from a 24-trial Bayesian (Optuna) search over latent-8
hyperparameters using **validation reconstruction quality**, with worst-case validation horizon as a robustness
tiebreak; a wider follow-up search did not displace it. Reconstruction is the appropriate criterion because the
model's role is to supply *coordinates*: each downstream forecaster learns its own dynamics in the latent space and
decodes predictions back through the frozen decoder, so what matters is how faithfully the encoder/decoder
round-trips the field, not how good the autoencoder's own SINDy equation is. The selected representation achieves a
full-series reconstruction MSE of 0.0066 and MAE of 0.062 and never collapses across validation windows, and it was
reproduced exactly inside the final package on the frozen simulation dataset **on its original same-stack GPU
profile** (a same-stack observation, re-confirmed by an optional provenance retrain; by default the shipped
representation is frozen/checksum-validated, not retrained, and exact reproduction is not a cross-hardware guarantee).

## 8. Notes and caveats

- **Do not** claim `trial_014` is the best **sparse SINDy dynamics** model. It is selected as a *representation* by
 reconstruction. Its own SINDy rollout horizon is modest — expected, and irrelevant to its role.
- **Do not** claim the forecasters use the autoencoder's SINDy equation. They use the **encoder, decoder and latent
 arrays** only; each learns its own latent dynamics.
- **Do** note the asymmetry: retraining the autoencoder requires a **GPU** (`device_used: cuda`), but latent-8
 *forecasting* is CPU-reproducible from the shipped export.
- **Do not** read the two objectives as aligned: at latent-8, best reconstruction and longest horizon **genuinely
 diverge**, which is precisely why representation and dynamics are selected by different criteria.
- **Do** treat the exact package reproduction as a **same-stack** result. By **default** this representation is
 **frozen/checksum-validated, not retrained** (2026-07-17 policy lock); the checkpoint reproduces bit-identically
 only on its original profile (RTX 2080 Ti, torch 2.6.0, seed 42, `deterministic=true`), re-confirmed by an
 optional opt-in provenance retrain (optional GPU retraining; see ../../../docs/advanced_latent8_gpu_retrain.md).
 A legitimate retrain on different hardware need not match bit-for-bit — the same nondeterminism that rejected the
 latent6 retrain — so exact reproduction is **not** a cross-hardware guarantee.
