# Advanced: latent-8 GPU retraining

This describes an optional, non-default workflow that retrains the latent-8 autoencoder on a
GPU. It exists to corroborate the provenance of the frozen representation, not to reproduce the
thesis results. The default reproduction never uses it.

## What it is for

The [latent-8 representation](ae_sindy.md) is a GPU-trained autoencoder. By default it is frozen
and validated by checksum, and everything downstream reproduces from the exported latents on
CPU. This advanced path retrains the autoencoder under the original recorded profile and
compares the result against the shipped checkpoint. On the original hardware and software stack
the retraining reproduced the checkpoint and exported latents exactly. That is corroborating
evidence for the frozen artifact; it is not a claim that retraining reproduces the checkpoint on
other hardware.

## Hardware and environment specificity

The retraining was tested with PyTorch 2.6.0 (built against CUDA 12.6, cuDNN 9.8, Triton 3.2.0)
on an NVIDIA GeForce RTX 2080 Ti, using the environment recorded in
[`environment-lock.yml`](../environment-lock.yml). It requires a PyTorch interpreter with a
working CUDA device; point the wrapper at it with `--torch-python` or `REPRO_TORCH_PYTHON` (see
[environment.md](environment.md)).

Chaotic-system and GPU nondeterminism mean a legitimate retraining on different hardware, a
different driver, or a different CUDA version need not match the frozen checkpoint bit for bit. A
non-match on other hardware would not invalidate the frozen result. Bitwise reproduction is
claimed only for the original same-stack profile.

## How it runs, and where it writes

The workflow is opt-in through the wrapper's advanced flag:

```bash
python scripts/reproduce_final_thesis.py --advanced-gpu-latent8-historical --plan
```

`--plan` shows what would happen and creates nothing. Without `--plan`, the wrapper builds a
scratch working area and, on the original cluster, submits a GPU batch job before comparing the
retrained artifacts against the package. It writes only under `test_repro_ae_sindy/`, which is
ignored by version control, and never modifies the package.

The batch-job settings used on the original cluster (partition, account, wall-clock limit, and
the RTX 2080 Ti resource request) are specific to that system and are included as an example.
They will need to be adapted to any other scheduler, and the path is only meaningful where a
matching GPU is available. This workflow is separate from, and not required by, the default
reproduction described in [validation.md](validation.md).
