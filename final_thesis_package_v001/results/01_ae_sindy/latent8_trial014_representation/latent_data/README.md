# Final reproduced latent dataset: repro_latent8_trial014

This folder holds the latent dataset produced by a reproduced AE-SINDy run. It provides the
coordinates in which the latent-8 forecasting models operate, allowing those models to be
reproduced without retraining the autoencoder.

Source run:
`final_thesis_package_v001/results/01_ae_sindy/latent8_trial014_representation/repro_latent8_trial014`

Source config:
`final_thesis_package_v001/configs/01_ae_sindy/latent8_trial014_representation.json`

Main files:

- `z_series.npy`: full latent trajectory
- `z_train.npy`: training latent data
- `z_valid.npy`: validation latent data
- `z_test.npy`: test latent data
- `u_reconstructed.npy`: decoded reconstruction in physical 64D space
- `reconstruction_summary.csv`: reconstruction statistics
- `source_model_info.json`: metadata for reproducibility

These arrays are the inputs for the downstream RC, NGRC, PINN and PERC forecasting experiments.
