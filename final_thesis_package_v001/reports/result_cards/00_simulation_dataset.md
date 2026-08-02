# Result Card — `simulation_dataset`

## 1. Result identity

The **frozen reference KSE trajectory** — the root dataset of the entire thesis. A 64-point spatial grid on a
domain of length L = 22.0, `dt = 0.1`, 30,606 solver steps with the first 500 discarded as transient, leaving
**30,106 post-transient steps** (~3,010 physical time units ≈ **130 Lyapunov times** at λ_max = 0.043).

**Thesis role: MAIN.** Every AE-SINDy and full64 forecasting result is defined against these exact bits, and it
is the **reproducibility boundary** of the package.

## 2. Inputs and dependencies

| item | path |
|---|---|
| config | [`configs/00_simulation/kse_simulation.json`](../../configs/00_simulation/kse_simulation.json) |
| data (package) | `results/00_simulation/u_series.npy` |
| locked source | `results/00_simulation/` (checksum-verified) |
| generator | `simulation/kse_simulation.py` |

**Dependencies: none.** This is the root of the chain. `u_series.npy` sha256 `85bb1902875e2b30…`.

## 3. Development and selection timeline

*Reconstructed from configs, config metadata, reports and on-disk artifacts. Where motivation is explicitly
documented it is cited; where it is inferred, the wording says so; where evidence is absent it says
"unknown from available evidence."*

| Step | Artifact/config/script | Step type | Candidate/search space | Metric or check used | Aim | Outcome / what was learned | Limitation or issue found | Why next step was needed | Connection to final result | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| **KSE solver run** | `final_thesis_package_v001/configs/00_simulation/kse_simulation.json` | simulation_generation | nx=64, L=22.0, dt=0.1, n_steps=30606, transient=500, seed=42 | solver completion + post-transient trajectory shape | Generate the reference KSE trajectory that every downstream result is defined against. | 30,106 post-transient steps (~3010 physical time, ~130 Lyapunov times at lambda_max=0.043) on a 64-point grid, L=22. | The trajectory is chaotic: a 1e-14 difference between FFT/BLAS implementations reaches attractor scale within ~25% of the span. | Bit-identical regeneration is impossible in principle, so the dataset had to be locked rather than rerun. | IS the root dataset: u_series.npy is the input to AE-SINDy and to every full64 forecasting model. | high |
| **Freeze decision** | `final_thesis_package_v001/results/00_simulation/u_series.npy` | freeze | n/a | sha256 85bb1902875e2b30... byte-identity at destination | Decide whether to regenerate the trajectory package-locally or freeze the locked dataset. | FROZEN (2026-07-14). The solver was NOT rerun; the locked dataset was copied and re-hashed bit-identical at the destination. | Reproduces the physics and statistics of the system, not the bits. This is a property of chaos, not a code defect. | With the dataset frozen, AE-SINDy latent-8 could be reproduced exactly (max abs difference 0.0) because it consumes these exact bits. | Defines the reproducibility BOUNDARY of the entire package. | high |
| **Package copy + diagnostic figure** | `final_thesis_package_v001/results/00_simulation/` | package_reproduction | n/a | sha256 of every copied array | Place the frozen dataset and its diagnostic figure inside the package. | Dataset + figure in package; figure kse_spatiotemporal_lyapunov.png (Lyapunov axis, emitted only when use_lyapunov=true). | The figure is a copy, not a package-side regeneration. | Downstream stages could then run entirely against package-local paths. | Supplies the one thesis-facing simulation figure and the data path in every downstream config. | high |

## 4. Final selected config/result

**Status: 🔒 FROZEN (Stage 2, 2026-07-14).** The solver was **not** rerun. The locked, checksum-verified dataset
was copied into `results/00_simulation/` and every array re-hashed **bit-identical at the destination**.

Config: `configs/00_simulation/kse_simulation.json` (`nx=64, L=22.0, dt=0.1, n_steps=30606, transient=500, seed=42`).

## 5. Final metrics

| quantity | value |
|---|---|
| post-transient steps | **30,106** |
| grid points (nx) | 64 |
| domain length L | 22.0 |
| dt | 0.1 (physical simulation time) |
| span | ~3,010 physical time ≈ **130 Lyapunov times** (λ_max = 0.043) |
| sha256 (`u_series.npy`) | `85bb1902875e2b30…` |

Splits used by everything downstream: **train `[0,20000)` / validation `[20000,25000)` / test `[25000,30106)`**.

## 6. Supporting plots

**Main result plot**
- `plots/00_simulation/kse_spatiotemporal_lyapunov.png` — the field u(x,t); x-axis **Lyapunov time**
 (`t [1/λ_max]`), y-axis space x. The `_lyapunov` filename suffix is emitted *only* when `use_lyapunov=true`,
 so the name itself certifies the axis.

**Selection/provenance plots** — none (nothing was selected; the dataset was generated once and frozen).
**Appendix/diagnostic plots** — none.

## 7. Thesis-facing statement

The Kuramoto–Sivashinsky field was integrated on a 64-point grid over a domain of length L = 22 with `dt = 0.1`,
discarding 500 transient steps and retaining 30,106 post-transient steps — roughly 130 Lyapunov times at
λ_max ≈ 0.043. This single trajectory is the reference dataset for the whole study: the autoencoder is trained on
it, the full-field forecasters predict it, and the latent forecasters predict its encoding. It is distributed as a
**frozen, checksum-verified artefact** rather than regenerated on demand, because bit-identical reproduction of a
chaotic trajectory is not achievable across environments — re-running the solver reproduces the *physics* and the
*statistics* of the system, not the *bits*.

## 8. Notes and caveats

- **Do not** claim the dataset is bit-reproducible. It is not, and this is a property of chaos, not a defect of the
 code: a 1e-14 difference between FFT implementations, library versions or thread counts is amplified to attractor
 scale within the first quarter of the trajectory.
- **Do not** describe the freeze as a shortcut. It is the reason AE-SINDy latent-8 reproduces **exactly**
 (max abs difference 0.0) — that stage consumes these exact bits.
- **Do** state that verification is by **checksum** (`85bb1902…`), not by re-running the solver.
- The figure is a **curated copy** of the run-native diagnostic, not a package-side regeneration.
