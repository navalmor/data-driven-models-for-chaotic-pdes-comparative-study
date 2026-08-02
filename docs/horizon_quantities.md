# Horizon and validity-time quantities

Three named quantities in this repository measure "how long a prediction stayed
usable". They look alike and were previously labelled alike, but they are not
interchangeable. This note records what each one is, where it is computed, and
exactly how the two pipelines differ.

**Nothing here changes a computed value.** The differences in §3 are documented
deliberately and left in place; changing them would alter published results and
requires its own approval.

---

## 1. The three quantities

| name in figures | code symbol | computed in | what it measures |
| --- | --- | --- | --- |
| **Prediction horizon** | `relative_l2_horizon_time` | `forecasting/src/forecasting/metrics.py` | time until a *forecast* of the 64-dimensional physical field exceeds the relative-\(L^2\) threshold |
| **Prediction horizon** | `sindy_valid_time` | `ae_sindy/ae_sindy/analysis.py` | time until the *decoded latent SINDy rollout* exceeds the same threshold against the same truth |
| **Reconstruction validity time** | `encoder_valid_time`, `encoder_rollout_valid_time` | `ae_sindy/ae_sindy/analysis.py` | time until the *autoencoder round trip* — encode then decode, **no dynamics** — exceeds the threshold |

The first two are the same construct and share the name. The third is a
different measurement: it evaluates representation fidelity, not forecasting,
and must never be called a prediction horizon.

### Which figure shows which

| figure | quantity | label |
| --- | --- | --- |
| `reconstruction.png` | `encoder_valid_time` | Reconstruction validity time |
| `rollout_encoder_comparison.png` | `encoder_rollout_valid_time` | Reconstruction validity time |
| `latent_space_comparison.png` | `sindy_valid_time` | Prediction horizon |
| `rollout_sindy_comparison.png` | `sindy_valid_time` | Prediction horizon |
| `*_relative_l2_horizon.png` | `relative_l2_horizon_time` | Prediction horizon |
| `*_spatiotemporal_comparison.png` | `relative_l2_horizon_time` | Prediction horizon |

---

## 2. What the two pipelines agree on

| aspect | forecasting | AE-SINDy | verdict |
| --- | --- | --- | --- |
| failure rule | `(~isfinite) \| (err > threshold)` | `(~isfinite) \| (err > threshold)` | identical |
| threshold | `relative_l2_threshold`, 0.5 in the shipped runs | `horizon_error_threshold`, 0.5 in both shipped configs | same value, different config key |
| error metric | \(\lVert y_t-\hat y_t\rVert_2 / (\lVert y_t\rVert_2 + \varepsilon)\), \(\varepsilon=10^{-12}\) | \(\lVert x_t-\hat x_t\rVert_2 / (\lVert x_t\rVert_2 + \varepsilon)\), \(\varepsilon=10^{-12}\) | identical, including \(\varepsilon\) |
| scoring space | 64-D physical field; latent cases decoded before scoring | `sindy_valid_time` scores the decoded rollout in the same 64-D field | identical for the forecast quantity |
| censoring recorded | `relative_l2_horizon_reached_end` | `threshold_crossed` | both record it |

---

## 3. Where they differ — documented, not changed

### 3.1 Index convention (one time step)

Let \(k\) be the first index at which the error series fails.

| pipeline | reported value | meaning |
| --- | --- | --- |
| forecasting | `horizon_steps = k`, time \(= k\,\Delta t\) | time of the **first failing** sample |
| AE-SINDy | `valid_index = k-1`, time \(= t[k-1]\) | time of the **last valid** sample |

The two therefore differ by exactly one \(\Delta t\) for the same error series.
Both are defensible; they are simply not the same convention.

### 3.2 Warm-up handling

| pipeline | initial-step handling |
| --- | --- |
| forecasting | `skip_initial = 50` in the shipped runs — the first 50 steps are seed/context values and are excluded from scoring |
| AE-SINDy | no equivalent; scoring starts at index 0 |

### 3.3 Censoring display

AE-SINDy prints \(\geq\) instead of \(=\) when the threshold was never crossed
inside the displayed window, because the value is then a lower bound rather than
an estimate. The forecasting plots record the same flag but do not display it.
Chapter 5 states that none of the eight forecasting horizons is right-censored,
so nothing renders differently today; adopting the same convention there would
be a safeguard for future data rather than a change to current figures.

---

## 4. Consequences for reading the thesis

- A **Prediction horizon** in Chapter 3 and one in Chapter 5 measure the same
  thing, but are not numerically comparable to within one \(\Delta t\), and the
  Chapter 5 values additionally exclude a 50-step warm-up.
- A **Reconstruction validity time** is not comparable to either. It bounds how
  long the autoencoder representation itself stays faithful, which is an upper
  bound on what any latent forecast could achieve, not a forecast result.
- Any text comparing a Chapter 3 validity time with a Chapter 5 horizon must say
  which convention it is using.

---

## 5. Notation

`T_h` is a physical simulation time and never appears on a Lyapunov-scaled axis.
`t_λ` is a position on a Lyapunov-scaled axis and never appears on a physical
one. The safety cutoff is `t_c` on a physical axis and `t_{λ,c}` on a
Lyapunov-scaled one. These are enforced by `common/plot_style.py` and asserted by
`tests/test_plot_style_conventions.py`.
