# AE-SINDy

AE-SINDy combines an autoencoder with SINDy (Sparse Identification of Nonlinear Dynamics). The
autoencoder learns a low-dimensional latent coordinate system for the KS field, and SINDy
looks for a sparse system of ordinary differential equations governing those latent
coordinates. In this repository the method plays two distinct roles, and it is important to
keep them apart.

The code is in [`ae_sindy/`](../ae_sindy/). The runnable entry points are
`ae_sindy/scripts/run_train.py`, `run_analysis.py`, and `run_visualize.py`.

## The latent-8 representation

The primary AE-SINDy result is the eight-dimensional representation
`latent8_trial014_representation`. It was selected for its reconstruction quality on the
validation split, as the best tier among the latent-8 candidates, with a robustness check that
it never collapses. Its full-series reconstruction error is a mean squared error of about
0.00659 and a mean absolute error of about 0.0620.

The point of this result is the coordinate system, not a dynamics claim. Four of the eight
forecasting models operate in these latent coordinates rather than on the physical field. They
use the frozen encoder, decoder, and exported latent trajectory; they do not use the
autoencoder's own SINDy equations. The representation should therefore be described as a good
learned coordinate system, not as the best sparse dynamics model.

Training the autoencoder uses a GPU. To keep the downstream work reproducible without one, the
repository ships the trained encoder and decoder and the exported latent arrays, so the four
latent-8 forecasting models reproduce on CPU from these exact bytes. The representation is
frozen and validated by checksum by default; it is not retrained during the default
reproduction. An optional GPU retraining path is described in
[advanced_latent8_gpu_retrain.md](advanced_latent8_gpu_retrain.md).

## The latent-6 dynamics baseline

The interpretability contribution is `latent6_trial002_dynamics_baseline`: a sparse set of 17
active quadratic terms discovered in a six-dimensional latent space. Evaluated on five fixed
held-out windows, its SINDy rollout horizons are 8.8, 2.4, 2.6, 7.5 and 2.1 in physical
simulation time — a mean of 4.68, with 8.8 the best of the five rather than a typical value.
`scripts/reproduce_latent6_multiwindow.py` regenerates that evaluation from the frozen
checkpoint; its outputs ship under
`final_thesis_package_v001/results/01_ae_sindy/latent6_trial002_dynamics_baseline/multiwindow_test_evaluation/`.

This result ships as a reanalysis of a frozen checkpoint, not a from-scratch retraining. A full
retraining was attempted and rejected because the checkpoint is not bit-reproducible across
environments (the same GPU nondeterminism discussed for the latent-8 representation). The
frozen model is reanalysed and visualised against the shipped simulation, which reproduces its
canonical numbers exactly. Two caveats belong with it: one latent coordinate carries no
above-threshold dynamics, and the discovered system captures the dominant quadratic couplings
rather than a complete latent model of the KS field. The short, window-dependent horizon is
characteristic of chaotic latent dynamics.

## The post-hoc predictive refit

An appendix result, `latent8_trial014_posthoc_thr0025_predictive`, asks how much autonomous
prediction the frozen latent-8 representation can support when the SINDy coefficients are refit
for prediction rather than for sparsity. With the autoencoder frozen, the coefficients are
re-solved on the training split, and a sparsity threshold is chosen by a multi-window
validation horizon. The resulting model reaches a mean test horizon of about 10.86, but it is
dense — roughly 81 percent of coefficients are active — so it is a predictive refit, not an
interpretable one. The forecasting models do not use these coefficients.

## Provenance note

The scientific method is that of Champion, Lusch, Kutz and Brunton, "Data-driven discovery of
coordinates and governing equations", PNAS 116(45):22445-22451, 2019.

The implementation basis is the public reference implementation released with that paper,
`kpchamp/SindyAutoencoders`, which is written in TensorFlow. The code in this repository is a
PyTorch reimplementation of it, adapted to the KS data pipeline, the global-scalar
normalisation, the analysis and rollout paths, the post-hoc refit, and the run diagnostics. The
upstream project is MIT-licensed and its notice is retained in
[`LICENSES/MIT-SindyAutoencoders.txt`](../LICENSES/MIT-SindyAutoencoders.txt). The exact
upstream revision used is not recorded in this repository's history.
