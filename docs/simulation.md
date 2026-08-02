# Simulation

The dataset underlying every result is a numerical solution of the one-dimensional
Kuramoto-Sivashinsky (KS) equation, a fourth-order nonlinear PDE whose solutions become
spatiotemporally chaotic on a large enough domain. This makes it a standard and demanding
test problem for data-driven models of chaotic dynamics.

## Solver

The solver is [`simulation/kse_simulation.py`](../simulation/kse_simulation.py). It integrates
the KS equation on a periodic domain using a spectral spatial discretisation and an
exponential time-differencing scheme, and writes the solution trajectory as a NumPy array
together with a small set of run parameters. It depends only on NumPy, SciPy, and Matplotlib.

The configuration used for the thesis dataset is recorded in
[`final_thesis_package_v001/configs/00_simulation/kse_simulation.json`](../final_thesis_package_v001/configs/00_simulation/kse_simulation.json).

## The frozen dataset

The authoritative trajectory lives in
[`final_thesis_package_v001/results/00_simulation/`](../final_thesis_package_v001/results/00_simulation/)
as `u_series.npy`. It has 30106 time steps and spans roughly 130 Lyapunov times. Every
autoencoder and forecasting result in the repository is defined against this exact array.

This dataset is frozen. The default reproduction validates it by checksum and does not
regenerate it. The reason is intrinsic to the problem: the KS equation is chaotic, so a
difference of the order of floating-point rounding between two environments — a different FFT
implementation, library version, or thread count — is amplified to the scale of the attractor
within a fraction of the trajectory. Re-running the solver reproduces the statistics and the
qualitative physics of the system, but not the exact values, and therefore not the exact
inputs the downstream models were fit to.

You can run the solver to generate a new trajectory for your own experiments. It will be a
valid KS solution with the same statistical character, but it will not be bit-identical to the
frozen dataset, and models fit against it will differ in their exact numbers.

The repository ships a single copy of the simulation data, under
`final_thesis_package_v001/results/00_simulation/`. There is no second copy.
