# Result Card — `ngrc_full64`

## 1. Result identity

**NGRC (Next-Generation Reservoir Computing) forecasting the full 64-dimensional physical KSE field.**
This is the **headline forecasting result of the thesis**: it predicts roughly **15× further** than any other model
in the study, and it is the only model whose accuracy is not bounded by a learned representation.

**Thesis role: MAIN.** NGRC is **deterministic** (no RNG), so its search *is* its selection procedure.

## 2. Inputs and dependencies

| item | path |
|---|---|
| package config | [`configs/02_forecasting/ngrc_full64.json`](../../configs/02_forecasting/ngrc_full64.json) |
| canonical source config | `full64_ngrc_selected_v003_stable_ridge.json` |
| result | `results/02_forecasting/ngrc/selected/ngrc_full64_selected_v003_stable_ridge/` |
| search source (survives) | `ngrc_full64_optimizer_search_v003_stable_ridge` |

**Depends on:** `results/00_simulation/u_series.npy` **only** — no autoencoder involved at any point.

## 3. Development and selection timeline

*Reconstructed from configs, config metadata, reports and on-disk artifacts. Where motivation is explicitly
documented it is cited; where it is inferred, the wording says so; where evidence is absent it says
"unknown from available evidence."*

| Step | Artifact/config/script | Step type | Candidate/search space | Metric or check used | Aim | Outcome / what was learned | Limitation or issue found | Why next step was needed | Connection to final result | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| **Smoke test** | `full64_ngrc_smoke_v002.json` | smoke | minimal execution, not a search | runner completes; data path resolves; outputs written | Check the runner, config schema and data path execute end-to-end before spending compute. | Pipeline executes for this model/data_mode. Available evidence suggests no scientific claim was drawn from this stage. | A smoke run says nothing about model quality. | A parameter feasibility probe was needed before any search. | Infrastructure only; no bearing on the final numbers. | medium |
| **Probes** | `full64_ngrc_probe_{bias,delay2,pairwise_small,trig}_v002.json (4 configs)` | probe | feature-library variants: bias term, delay depth 2, small pairwise (quadratic) set, trigonometric features | validation relative_l2_horizon_time | Test which NGRC FEATURE LIBRARY components matter before searching continuous hyperparameters. Configuration differences indicate the four probes isolate one library component each. | The final config keeps bias and pairwise (quadratic) features and uses sin/cos - available evidence suggests the probes settled the library structure rather than the ridge/context values. | Probes fix structure but cannot tune continuous parameters (context length, ridge). | A grid then an optimizer search were needed over the continuous parameters. | Fixed the structural choices the later searches held constant. | medium |
| **Grid search v002** | `full64_ngrc_grid_search_v002.json` | grid_search | coarse grid over NGRC readout/context parameters (exact ranges: see config) | validation relative_l2_horizon_time | Find the promising region coarsely before spending an optimizer budget. | Region-finding only. NOTE: the grid OUTPUT directory no longer exists on disk (removed in cleanup Wave 1), so this step is evidenced by its config, not its results. | Grid outputs deleted; the exact best grid point is unknown from available evidence. | The latent8 sibling config records the policy 'optimizer_only_after_grid_region_finding', indicating the grid was used to bound the optimizer, not to select. | Bounded the optimizer search; not itself selection evidence. | medium |
| **Optimizer search v002** | `full64_ngrc_optimizer_search_v002.json (+ ..._tree_search_v002.json)` | optimizer_search | optimizers [gp, dummy]; continuous NGRC space incl. ridge_alpha with NO enforced lower bound | validation relative_l2_horizon_time (maximize) | Select NGRC full64 by continuous validation-only search. | Produced selected v002: gp-sourced, feature dimension 1665, validation horizon 419.2, ridge_alpha 6.3e-10. | DECISIVE DEFECT: the winning ridge_alpha (6.3e-10) is far below the well-posed range. The 1665 delay-embedded quadratic features are strongly collinear, so a near-zero ridge leaves the solution partly determined by floating-point reduction order and the validation horizon is not reproducible across BLAS thread configurations. | The search had found a peak in a NUISANCE parameter (floating-point noise), not a better model - so the ridge range had to be corrected and the search rerun. | SUPERSEDED. Its 419.2 is not a defensible number; it is the reason the stable-ridge stage exists. | high |
| **Reproducibility audit** | `v002_reproducibility_audit.md` | documentation | n/a | structural config match + numerical reproducibility across thread configurations | Check NGRC v002 reproducibility after the RC spectral-radius issue was found. | NGRC v002 passes the STRUCTURAL audit (selected config matches its validation-selected source exactly; NGRC has no random reservoir). The NUMERICAL problem is the ill-conditioned ridge, not a config mismatch. | Structural correctness does not imply numerical stability. | Motivated a corrected search with a bounded ridge range. | Establishes that the v002 defect is numerical conditioning, not provenance. | high |
| **Optimizer search v003** | `full64_ngrc_optimizer_search_v003.json` | optimizer_search | v003 space (superseded) | validation relative_l2_horizon_time | Intermediate corrected search. | Superseded by v003_stable_ridge; the stable-ridge config's metadata records 'supersedes: full64_ngrc_optimizer_search_v003.json'. Its output directory does not exist on disk. | What v003 changed relative to v002, and why it in turn needed replacing, is unknown from available evidence beyond the supersedes pointer. | The stable-ridge variant is the one that ran and produced the final config. | Superseded intermediate. | low |
| **Optimizer search v003_stable_ridge - SOURCE** | `full64_ngrc_optimizer_search_v003_stable_ridge.json` | optimizer_search | ridge_alpha restricted to [1e-6, 1e-2] (corrected range); optimizers incl. forest; validation-only (evaluate_test=false, append_registry=false enforced in optimizer_search_ngrc.py) | validation relative_l2_horizon_time (maximize) | Re-run the search with the ridge lower-bounded into the well-posed regime. | forest trial 108 won: context_steps 18, sin+cos on, ridge_alpha 2.62e-05, feature dimension 4609. The selected value sits an order of magnitude above the lower bound. | The honest number is LOWER than the superseded one (379.9 vs 419.2). The correction costs headline performance. | n/a - produced the final config. | IS the source of the final selected config. | high |
| **Selected v003_stable_ridge** | `full64_ngrc_selected_v003_stable_ridge.json` | selected_evaluation | n/a (locked config) | validation horizon for selection; test evaluated once after locking (test_used_for_selection: false) | Lock the stable-ridge config and report test once. | valid 379.9 / test 324.0. Stability note: same-profile deterministic rerun bit-identical; pinned/unpinned validation horizon difference 0.29%. | NGRC full64 is deterministic (no RNG), so no seed distribution exists or is definable for it. | n/a - final. | IS the final selected config. | high |
| **Package reproduction (Stage 4)** | `final_thesis_package_v001/configs/02_forecasting/ngrc_full64.json` | package_reproduction | n/a (single locked config) | validation/test relative_l2_horizon_time vs canonical | Re-execute the locked selected config inside the package against package-local data. | Reproduced canonical to the decimal on the conda torch interpreter forced to CPU (CUDA_VISIBLE_DEVICES=''). Registry unchanged. | none identified at this step | n/a - accepted final form. | IS the package result: results/02_forecasting/ngrc/selected/ngrc_full64_selected_v003_stable_ridge | high |
| **Figures (Stage 6)** | `final_thesis_package_v001/plots/02_forecasting/per_model/ngrc_full64/ (4 PNGs)` | plotting | n/a | n/a | Generate the per-model forecasting figures from the accepted package results. | 4 PNGs: {validation,test}_{relative_l2_horizon,spatiotemporal_comparison}. Lyapunov time axes (use_lyapunov=true, lambda_max=0.043). | Figure time axes are in Lyapunov units, so horizon markers do NOT equal the physical-time tables. | n/a - terminal step. | Supplies this model's thesis-facing figures. | high |

## 4. Final selected config/result

**Package config:** `configs/02_forecasting/ngrc_full64.json` (`run_id: ngrc_full64_selected_v003_stable_ridge`)
**Canonical source:** `full64_ngrc_selected_v003_stable_ridge.json`
**Result:** `results/02_forecasting/ngrc/selected/ngrc_full64_selected_v003_stable_ridge/`

Selected hyperparameters: `context_steps 18`, `include_sin`/`include_cos` **true**, `ridge_alpha` **2.62e-05**,
feature dimension **4609**; from **forest trial 108** of the stable-ridge search.
`test_used_for_selection: false`. **Supersedes** `full64_ngrc_selected_v002.json`.

## 5. Final metrics

> **Units.** Validation and test horizons are **physical simulation time**. The per-model forecasting
> figures plot **Lyapunov time** (`t·λ_max`, λ_max = 0.043), so a horizon marker on those axes equals
> `physical × 0.043` and will **not** match the numbers below. **Quote this table, never the axes.**

| metric | value |
|---|---|
| **validation horizon** | **379.9** (physical simulation time) |
| **test horizon** | **324.0** (physical simulation time) |
| seed | n/a — **deterministic**, no RNG |
| stability | same-profile deterministic rerun **bit-identical**; pinned/unpinned validation horizon difference **0.29 %** |

For scale: 324.0 physical time ≈ 13.9 Lyapunov times, against ~0.4–1.7 for every other model.

## 6. Supporting plots

**Main result plots** (`plots/02_forecasting/per_model/ngrc_full64/`)
- `test_relative_l2_horizon.png`, `test_spatiotemporal_comparison.png` — the headline evidence.

**Selection/provenance plots** (`plots/optimizer_search/ngrc/ngrc_full64/`)
- `optimizer_convergence.png` — **genuine selection evidence**, because NGRC is deterministic: its search *is* the
 selection procedure. The strongest appendix candidate of the whole optimizer set.
- `optimizer_sampling_matrix.png`, `optimizer_surrogate_landscape.png` — appendix only if coverage is challenged.

**Appendix/diagnostic plots**
- `validation_relative_l2_horizon.png`, `validation_spatiotemporal_comparison.png` — selection split, not performance.

*Note: the optimiser figures can be regenerated from the compact public trial-history dataset under
`forecasting/data/optimizer_search/`; identical PNG checksums across environments are not guaranteed.*

## 7. Thesis-facing statement

NGRC on the full 64-dimensional field is the strongest forecasting result in the study, reaching a validation
horizon of 379.9 and a held-out test horizon of 324.0 in physical simulation time — roughly fifteen times further
than any other model tested. Because NGRC has no random component, its validation-only optimizer search *is* its
selection procedure, and the selected configuration is reproducible bit-for-bit under a fixed runtime profile. The
final configuration is deliberately **not** the highest-scoring one ever found. An earlier selection (v002) reported
419.2, but it had been awarded a ridge coefficient of 6.3e-10: with 1665 strongly collinear delay-embedded quadratic
features, a near-zero ridge leaves the readout solve ill-conditioned, so the solution is partly determined by
floating-point reduction order and the horizon is not reproducible across BLAS thread configurations. The search had
found a peak in a **numerical nuisance parameter**, not a better model. Restricting the ridge range to the well-posed
regime [1e-6, 1e-2] and re-running the search produced the locked configuration (ridge 2.62e-05), whose **lower**
379.9 is the defensible number.

## 8. Notes and caveats

- **Do not** quote **419.2**. It is superseded and was partly a floating-point artefact of an ill-conditioned solve.
- **Do** state plainly that the correction **lowered** the headline number. That is what makes it credible.
- **Do not** describe NGRC full64 as seed-selected or seed-robust. It is **deterministic**: no RNG, no seed
 distribution, and **no seed-robustness boxplot exists or is definable** for it.
- **Do not** compare the 324.0 test horizon to latent-8 models as if the methods were the only difference — full64
 sees the true field, latent8 sees a compressed reconstruction of it.
- Its selection-process figure question is **deferred**; the existing `optimizer_convergence.png` already
 serves as its selection evidence.
- The optimiser figures regenerate from the public dataset described above; they remain appendix
 diagnostics, not performance evidence.
