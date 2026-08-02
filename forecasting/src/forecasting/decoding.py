from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)

Array = np.ndarray


@dataclass(frozen=True)
class DecodeResult:
    """
    Metadata returned after decoding a latent trajectory to physical space.
    """

    latent_input_path: Path
    physical_output_path: Path
    latent_shape: tuple[int, int]
    physical_shape: tuple[int, int]


def _resolve_path(repo_root: Path, path: str | Path) -> Path:
    """
    Resolve a path relative to repo_root unless it is already absolute.
    """
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _require_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer, got bool.")

    value = int(value)

    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")

    return value


def _load_npy_2d(path: Path, *, name: str) -> Array:
    """
    Load a 2D NumPy array from disk.
    """
    if not path.exists():
        raise FileNotFoundError(f"{name} file does not exist: {path}")

    arr = np.load(path, allow_pickle=False)

    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D array with shape (time_steps, dimension), "
            f"got shape {arr.shape} at {path}."
        )

    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one timestep: {path}")

    if arr.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one state dimension: {path}")

    return np.asarray(arr, dtype=np.float64)



def _decoder_subprocess_env(repo_root: Path) -> dict[str, str]:
    """
    Build a subprocess environment for the AE-SINDy decoder.

    The codec script imports:
        from ae_sindy.model import SINDyAutoencoderModel

    In this repository, that package is available through:
        <repo_root>/ae_sindy

    Passing PYTHONPATH explicitly makes latent decoding independent of the
    user's interactive shell, working directory, or HPC module state.
    """
    env = os.environ.copy()

    candidate_paths = [
        repo_root / "ae_sindy",
        repo_root,
    ]

    existing = env.get("PYTHONPATH", "")
    for part in existing.split(os.pathsep):
        if part:
            candidate_paths.append(Path(part))

    unique_paths: list[str] = []
    seen: set[str] = set()

    for path in candidate_paths:
        path_str = str(path)
        if path_str not in seen:
            unique_paths.append(path_str)
            seen.add(path_str)

    env["PYTHONPATH"] = os.pathsep.join(unique_paths)
    return env


def decode_latent_file_to_physical_file(
    *,
    repo_root: str | Path,
    decoder_run_dir: str | Path,
    latent_input_path: str | Path,
    physical_output_path: str | Path,
    expected_latent_dim: int,
    expected_physical_dim: int,
) -> DecodeResult:
    """
    Decode one latent trajectory file into the original physical KSE space.

    This function is a strict wrapper around:
        scripts/ae_sindy_latent_codec.py decode

    It does not decide rollout safety or forecast quality. The runner should
    pass only a finite safe latent prefix to this function.
    """
    repo_root = Path(repo_root).expanduser().resolve()

    expected_latent_dim = _require_positive_int(
        expected_latent_dim,
        name="expected_latent_dim",
    )
    expected_physical_dim = _require_positive_int(
        expected_physical_dim,
        name="expected_physical_dim",
    )

    codec_script = repo_root / "scripts" / "ae_sindy_latent_codec.py"
    decoder_run_dir = _resolve_path(repo_root, decoder_run_dir)
    latent_input_path = _resolve_path(repo_root, latent_input_path)
    physical_output_path = _resolve_path(repo_root, physical_output_path)

    if not repo_root.exists():
        raise FileNotFoundError(f"Repository root does not exist: {repo_root}")

    if not codec_script.exists():
        raise FileNotFoundError(f"AE-SINDy codec script does not exist: {codec_script}")

    if not decoder_run_dir.exists():
        raise FileNotFoundError(f"Decoder run directory does not exist: {decoder_run_dir}")

    latent = _load_npy_2d(latent_input_path, name="latent input")

    if latent.shape[1] != expected_latent_dim:
        raise ValueError(
            f"Latent input dimension mismatch: expected {expected_latent_dim}, "
            f"got {latent.shape[1]} at {latent_input_path}."
        )

    if not np.all(np.isfinite(latent)):
        raise ValueError(
            f"Latent input contains NaN or Inf. Decode only a finite safe prefix: "
            f"{latent_input_path}"
        )

    physical_output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(codec_script),
        "decode",
        "--run-dir",
        str(decoder_run_dir),
        "--input",
        str(latent_input_path),
        "--output",
        str(physical_output_path),
    ]

    logger.debug(
        "Starting latent decode | latent=%s decoder=%s output=%s latent_shape=%s",
        latent_input_path,
        decoder_run_dir,
        physical_output_path,
        tuple(latent.shape),
    )

    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=_decoder_subprocess_env(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )

    if completed.stdout.strip():
        logger.debug("Decoder stdout | %s", completed.stdout.strip())

    if completed.stderr.strip():
        logger.debug("Decoder stderr | %s", completed.stderr.strip())

    if completed.returncode != 0:
        raise RuntimeError(
            "AE-SINDy latent decoding failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Return code: {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    physical = _load_npy_2d(
        physical_output_path,
        name="decoded physical output",
    )

    if physical.shape[0] != latent.shape[0]:
        raise ValueError(
            f"Decoded timestep count mismatch: latent has {latent.shape[0]} steps, "
            f"physical output has {physical.shape[0]} steps."
        )

    if physical.shape[1] != expected_physical_dim:
        raise ValueError(
            f"Decoded physical dimension mismatch: expected {expected_physical_dim}, "
            f"got {physical.shape[1]} at {physical_output_path}."
        )

    if not np.all(np.isfinite(physical)):
        raise ValueError(
            f"Decoded physical output contains NaN or Inf: {physical_output_path}"
        )

    result = DecodeResult(
        latent_input_path=latent_input_path,
        physical_output_path=physical_output_path,
        latent_shape=(int(latent.shape[0]), int(latent.shape[1])),
        physical_shape=(int(physical.shape[0]), int(physical.shape[1])),
    )

    logger.debug(
        "Finished latent decode | latent_shape=%s physical_shape=%s output=%s",
        result.latent_shape,
        result.physical_shape,
        result.physical_output_path,
    )

    return result
