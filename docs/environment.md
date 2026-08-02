# Environment

This project runs on Python 3.12. There are two ways to set it up: a standard virtual
environment for the default CPU workflow, and a recorded Conda environment that documents
the exact stack used to produce the tested GPU checkpoints.

## Standard installation (recommended)

The default reproduction, validation, plotting, simulation, and optimiser examples run on
CPU and need only the packages in [`requirements.txt`](../requirements.txt).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` lists pinned direct dependencies, not a full transitive lock:

```
numpy==1.26.4
scipy==1.17.1
scikit-learn==1.3.2
scikit-optimize==0.10.2
matplotlib==3.10.8
torch==2.6.0
```

After installation, a quick check:

```bash
python -c "import numpy, scipy, sklearn, skopt, matplotlib, torch; \
print(numpy.__version__, scipy.__version__, sklearn.__version__, \
matplotlib.__version__, skopt.__version__, torch.__version__)"
```

which should report `1.26.4 1.17.1 1.3.2 3.10.8 0.10.2 2.6.0`.

## A note on PyTorch

The packaged model checkpoints were built with the Conda package
`pytorch=2.6.0=cuda126_mkl_py312_h30b5a27_304`. Installing `torch==2.6.0` from PyPI gives a
different build with a different BLAS and CUDA configuration. That build runs the default
reproduction workflow on CPU and should pass the numeric-tolerance validation contract, but
it is not guaranteed to reproduce the checkpoints bit for bit. If you need a specific
CPU-only or CUDA build of PyTorch, follow the official PyTorch installation instructions for
your platform rather than relying on the pin above.

The default reproduction does not require a GPU. The reproduction wrapper runs the PyTorch
steps (latent decoding, the PINN, and AE-SINDy analysis) on CPU.

## The tested GPU environment

[`environment-lock.yml`](../environment-lock.yml) records the environment used for the GPU
work, including the optional autoencoder retraining. It has two layers:

- a Conda layer that provides PyTorch and the CUDA stack, and
- a `pip` subsection that provides the scientific stack (numpy, scipy, scikit-learn,
  scikit-optimize, matplotlib).

Recreate it with:

```bash
conda env create -f environment-lock.yml
```

**NumPy version.** In the tested environment the scientific packages were installed with pip
on top of the Conda environment, and pip pinned NumPy to 1.26.4. This overrides the Conda
NumPy 2.2.5, so 1.26.4 is the effective, validated version. The lock file keeps both entries
because that is what was actually installed; when the environment is rebuilt from the lock,
the pip layer restores NumPy 1.26.4 as the effective version. The standard `requirements.txt`
installation uses NumPy 1.26.4 directly and avoids this subtlety.

## GPU stack details

The GPU workflow was tested with:

| component | value |
|---|---|
| PyTorch | 2.6.0, built against CUDA 12.6 (`torch.version.cuda`) |
| CUDA toolkit (Conda packages) | 12.8 |
| cuDNN | 9.8 |
| Triton | 3.2.0 |
| GPU | NVIDIA GeForce RTX 2080 Ti |

The PyTorch build CUDA version (12.6) and the Conda CUDA toolkit packages (12.8) are distinct
and are reported separately on purpose. The NVIDIA driver version depends on the host and is
not asserted here. Results on other GPUs, drivers, or CUDA versions are not guaranteed to be
bitwise identical; see [validation.md](validation.md) for what the reproduction contract does
and does not promise.

## Selecting interpreters for the reproduction wrapper

`scripts/reproduce_final_thesis.py` uses two interpreters: one for the pure NumPy/SciPy steps
and one that provides PyTorch. A single virtual environment created from `requirements.txt`
can serve both roles, which is the normal case off the original cluster. You can point the
wrapper at specific interpreters when needed:

- `--system-python PATH` or `REPRO_SYSTEM_PYTHON` for the NumPy/SciPy steps,
- `--torch-python PATH` or `REPRO_TORCH_PYTHON` for the PyTorch steps.

If neither is given, the wrapper uses the original cluster paths when they exist, and
otherwise falls back to the interpreter running the script (requiring that it can import
`torch` for steps that need it). If no PyTorch-capable interpreter is found for a step that
needs one, the wrapper stops with a message explaining which option to set.
