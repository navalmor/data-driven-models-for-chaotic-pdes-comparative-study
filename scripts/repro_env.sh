#!/usr/bin/env bash
# Deterministic reproduction profile (H-03).
#
#   source scripts/repro_env.sh
#
# MUST be sourced BEFORE python starts. These variables are read when the
# interpreter, the BLAS libraries, and the cuBLAS handle are initialised; setting
# them from inside Python (os.environ) is too late to have any effect.
#
# Two different BLAS backends are in play in this environment and BOTH must be
# pinned: NumPy/SciPy use OpenBLAS (-> OPENBLAS_NUM_THREADS), PyTorch is an
# MKL-backed build (-> MKL_NUM_THREADS). See docs/environment.md.
#
# This script only exports environment variables. It runs no experiments.
#
# Pinning both backends fixes the floating-point reduction order, so a final selected
# configuration reproduces bit-identically on a fixed software stack.

# --- single-threaded linear algebra (fixes float reduction order) ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# --- interpreter + CUDA determinism ---
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "===== deterministic reproduction profile active (H-03) ====="
echo "  OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "  MKL_NUM_THREADS=${MKL_NUM_THREADS}"
echo "  OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
echo "  NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS}"
echo "  VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS}"
echo "  PYTHONHASHSEED=${PYTHONHASHSEED}"
echo "  CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}"
echo "  (source this BEFORE launching python; run final selected configs only)"
echo "============================================================"
