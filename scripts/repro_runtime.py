"""In-process deterministic runtime controls (H-03).

Companion to ``scripts/repro_env.sh``. The shell profile pins the environment
variables that must exist *before* the interpreter starts (thread counts,
PYTHONHASHSEED, CUBLAS_WORKSPACE_CONFIG); this module pins what can only be set
*inside* the process: RNG seeds, torch thread counts, and deterministic kernels.

Sourcing ``repro_env.sh`` remains the primary mechanism. ``apply_repro_runtime``
is a defence-in-depth layer so a run is still deterministic if a script is
launched without the wrapper.

The point of all three controls is reduction order. Unseeded generators, kernels
that split work across a variable number of threads, and autotuned cuDNN
algorithms each let the same configuration accumulate floating-point results in a
different order, so repeated runs disagree in the last significant digits. Fixing
them is what allows a selected configuration to be re-trained and re-evaluated to
the same numbers on a fixed software stack.

This module is not imported by the runners. The shipped results were produced
with ``repro_env.sh`` sourced, and that remains the supported route; this helper
is available to anyone driving the code from their own script, where sourcing a
shell profile is not possible.

Usage:
    from repro_runtime import apply_repro_runtime
    apply_repro_runtime(seed=0)
"""

from __future__ import annotations

import logging
import os
import random

logger = logging.getLogger("repro.runtime")

REPRO_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "PYTHONHASHSEED",
    "CUBLAS_WORKSPACE_CONFIG",
)


def check_repro_env(verbose: bool = True) -> dict[str, str | None]:
    """Report whether scripts/repro_env.sh was sourced before the interpreter started.

    These variables cannot be fixed from here: by the time Python is running it is
    too late for them to take effect. This only tells you whether the launch was
    correct.
    """
    seen = {name: os.environ.get(name) for name in REPRO_ENV_VARS}
    if verbose:
        missing = [name for name, value in seen.items() if value is None]
        if missing:
            logger.warning(
                "Deterministic launch profile not fully active; missing %s. "
                "Source scripts/repro_env.sh BEFORE python to guarantee byte-identical results.",
                ", ".join(missing),
            )
        else:
            logger.info("Deterministic launch profile active: %s", seen)
    return seen


def apply_repro_runtime(
    seed: int = 0,
    deterministic: bool = True,
    warn_only: bool = True,
) -> None:
    """Seed every RNG and force single-threaded, deterministic kernels.

    Args:
        seed: seed for ``random``, NumPy, and torch (CPU + CUDA).
        deterministic: enable deterministic torch algorithms and cuDNN.
        warn_only: pass through to ``torch.use_deterministic_algorithms``. Kept
            True by default because a few ops have no deterministic kernel in
            torch 2.6 and would otherwise raise mid-run.

    Safe to call when torch (or NumPy) is not installed.
    """
    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Raises if the interop pool is already initialised (i.e. torch has
        # already run parallel work). Not fatal: the value cannot be changed
        # after startup, so honour whatever was set at launch.
        pass

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from common.logging_setup import configure_logging

    configure_logging("INFO", component="REPRO")
    check_repro_env()
    apply_repro_runtime()
    logger.info("apply_repro_runtime(): OK")
