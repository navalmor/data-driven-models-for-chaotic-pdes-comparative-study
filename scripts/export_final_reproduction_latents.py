from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _bootstrap_path in (_REPO_ROOT, _REPO_ROOT / "ae_sindy"):
    if str(_bootstrap_path) not in sys.path:
        sys.path.insert(0, str(_bootstrap_path))

import numpy as np
import torch

from common.logging_setup import add_logging_arguments, configure_from_args, get_logger
from ae_sindy.model import SINDyAutoencoderModel

logger = get_logger("ae_sindy.export_latents")


def load_params(run_dir: Path) -> Any:
    params_path = run_dir / "params.pkl"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing params.pkl: {params_path}")
    with params_path.open("rb") as f:
        return pickle.load(f)


def load_state_dict(run_dir: Path) -> dict:
    ckpt_path = run_dir / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing model.pt: {ckpt_path}")

    obj = torch.load(ckpt_path, map_location="cpu")

    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]

    return obj


def get_param(params: Any, key: str, default=None):
    if isinstance(params, dict):
        return params.get(key, default)
    return getattr(params, key, default)


def load_normalization(run_dir: Path):
    path = run_dir / "normalization_stats.json"
    if not path.exists():
        return np.float32(0.0), np.float32(1.0)

    stats = json.loads(path.read_text())
    mean = stats.get("mean", stats.get("data_mean", 0.0))
    std = stats.get("std", stats.get("data_std", 1.0))
    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


def iter_named_tensors(obj: Any, prefix: str = "") -> Iterable[Tuple[str, torch.Tensor]]:
    """Recursively collect tensors from dict/list/tuple outputs."""
    if torch.is_tensor(obj):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            name = f"{prefix}.{k}" if prefix else str(k)
            yield from iter_named_tensors(v, name)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            name = f"{prefix}.{i}" if prefix else str(i)
            yield from iter_named_tensors(v, name)


def choose_tensor_by_keys(
    tensors: Dict[str, torch.Tensor],
    preferred_keys: list[str],
    batch_size: int,
    width: int,
) -> torch.Tensor | None:
    lower_map = {k.lower(): k for k in tensors.keys()}

    for key in preferred_keys:
        key_lower = key.lower()
        for existing_lower, original_key in lower_map.items():
            if existing_lower == key_lower or existing_lower.endswith("." + key_lower):
                t = tensors[original_key]
                if t.ndim == 2 and t.shape[0] == batch_size and t.shape[1] == width:
                    return t

    for key in preferred_keys:
        key_lower = key.lower()
        for existing_lower, original_key in lower_map.items():
            if key_lower in existing_lower:
                t = tensors[original_key]
                if t.ndim == 2 and t.shape[0] == batch_size and t.shape[1] == width:
                    return t

    return None


def encode_decode_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    latent_dim: int,
    input_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return z and reconstructed x from the trained AE-SINDy model.

    The project model forward requires x and dx, so for pure latent export we pass
    dx=zeros_like(x). This is safe because z=E(x) and x_hat=D(z) do not require
    true dx.
    """

    batch_size = x.shape[0]

    # First try direct encoder/decoder style APIs.
    possible_encoder_names = ["encode", "encoder"]
    possible_decoder_names = ["decode", "decoder"]

    encoder = None
    decoder = None

    for name in possible_encoder_names:
        if hasattr(model, name):
            encoder = getattr(model, name)
            break

    for name in possible_decoder_names:
        if hasattr(model, name):
            decoder = getattr(model, name)
            break

    if encoder is not None and decoder is not None:
        z = encoder(x)
        x_hat = decoder(z)
        return z, x_hat

    # Otherwise use the AE-SINDy forward call.
    dx = torch.zeros_like(x)

    try:
        out = model(x, dx)
    except TypeError:
        out = model(x)

    tensors = {name: tensor for name, tensor in iter_named_tensors(out)}

    # Useful debug if extraction fails.
    if not tensors:
        raise RuntimeError(f"Model output has no tensors. Output type: {type(out)}")

    z = choose_tensor_by_keys(
        tensors,
        preferred_keys=[
            "z",
            "z_latent",
            "latent",
            "encoded",
            "z0",
            "z_t",
        ],
        batch_size=batch_size,
        width=latent_dim,
    )

    x_hat = choose_tensor_by_keys(
        tensors,
        preferred_keys=[
            "x_decode",
            "x_decoded",
            "x_hat",
            "x_recon",
            "x_reconstructed",
            "reconstruction",
            "decoded",
            "x_pred",
        ],
        batch_size=batch_size,
        width=input_dim,
    )

    # Shape-based fallback.
    if z is None:
        latent_candidates = [
            (name, t)
            for name, t in tensors.items()
            if t.ndim == 2 and t.shape[0] == batch_size and t.shape[1] == latent_dim
        ]
        if latent_candidates:
            z = latent_candidates[0][1]

    if x_hat is None:
        input_candidates = [
            (name, t)
            for name, t in tensors.items()
            if t.ndim == 2 and t.shape[0] == batch_size and t.shape[1] == input_dim
        ]
        if input_candidates:
            x_hat = input_candidates[0][1]

    if z is None or x_hat is None:
        logger.debug("Available model output tensors:")
        for name, t in tensors.items():
            logger.debug("  %s: shape=%s", name, tuple(t.shape))
        raise RuntimeError(
            f"Could not extract z and reconstruction. "
            f"Needed latent_dim={latent_dim}, input_dim={input_dim}."
        )

    return z, x_hat


def main():
    parser = argparse.ArgumentParser(
        description="Export final reproduced AE-SINDy latent datasets for downstream RC/NGRC/PINN/PERC."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--data-path", default="simulation/simulation_data/u_series.npy")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-end", type=int, default=20000)
    parser.add_argument("--valid-end", type=int, default=25000)
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_from_args(args, component="AE-SINDy")

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    params = load_params(run_dir)

    if isinstance(params, dict):
        params = dict(params)
        params["device"] = torch.device(device)

    input_dim = int(get_param(params, "input_dim", 64))
    latent_dim = int(get_param(params, "latent_dim"))

    model = SINDyAutoencoderModel(params).to(device)
    state_dict = load_state_dict(run_dir)
    model.load_state_dict(state_dict)
    model.eval()

    u = np.load(args.data_path).astype(np.float32)
    mean, std = load_normalization(run_dir)

    normalize = bool(get_param(params, "normalize", True))

    if normalize:
        u_model = (u - mean) / std
    else:
        u_model = u.copy()

    z_batches = []
    uhat_batches = []

    with torch.no_grad():
        for start in range(0, len(u_model), args.batch_size):
            batch_np = u_model[start:start + args.batch_size]
            batch = torch.from_numpy(batch_np).to(device)

            z, u_hat_model = encode_decode_model(
                model=model,
                x=batch,
                latent_dim=latent_dim,
                input_dim=input_dim,
            )

            z_batches.append(z.detach().cpu().numpy())
            uhat_batches.append(u_hat_model.detach().cpu().numpy())

    z_series = np.concatenate(z_batches, axis=0).astype(np.float32)
    u_hat_model = np.concatenate(uhat_batches, axis=0).astype(np.float32)

    if normalize:
        u_reconstructed = u_hat_model * std + mean
    else:
        u_reconstructed = u_hat_model

    u_reconstructed = u_reconstructed.astype(np.float32)

    mse_per_timestep = np.mean((u_reconstructed - u) ** 2, axis=1).astype(np.float32)
    mae_per_timestep = np.mean(np.abs(u_reconstructed - u), axis=1).astype(np.float32)

    train_end = args.train_end
    valid_end = args.valid_end
    test_end = len(u)

    np.save(output_dir / "z_series.npy", z_series)
    np.save(output_dir / "u_reconstructed.npy", u_reconstructed)
    np.save(output_dir / "reconstruction_mse_per_timestep.npy", mse_per_timestep)
    np.save(output_dir / "reconstruction_mae_per_timestep.npy", mae_per_timestep)

    np.save(output_dir / "z_train.npy", z_series[:train_end])
    np.save(output_dir / "z_valid.npy", z_series[train_end:valid_end])
    np.save(output_dir / "z_test.npy", z_series[valid_end:test_end])

    summary_path = output_dir / "reconstruction_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "n_samples",
                "input_dim",
                "latent_dim",
                "mse_mean",
                "mse_min",
                "mse_max",
                "mse_std",
                "mae_mean",
                "mae_min",
                "mae_max",
                "mae_std",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "label": args.label,
            "n_samples": len(u),
            "input_dim": u.shape[1],
            "latent_dim": z_series.shape[1],
            "mse_mean": float(np.mean(mse_per_timestep)),
            "mse_min": float(np.min(mse_per_timestep)),
            "mse_max": float(np.max(mse_per_timestep)),
            "mse_std": float(np.std(mse_per_timestep)),
            "mae_mean": float(np.mean(mae_per_timestep)),
            "mae_min": float(np.min(mae_per_timestep)),
            "mae_max": float(np.max(mae_per_timestep)),
            "mae_std": float(np.std(mae_per_timestep)),
        })

    metadata = {
        "label": args.label,
        "source_run_dir": str(run_dir),
        "source_config": args.source_config,
        "source_data_path": args.data_path,
        "normalization_stats_path": str(run_dir / "normalization_stats.json"),
        "model_checkpoint_path": str(run_dir / "model.pt"),
        "params_path": str(run_dir / "params.pkl"),
        "normalize": normalize,
        "device_used": device,
        "input_shape": list(u.shape),
        "latent_shape": list(z_series.shape),
        "splits": {
            "train": [0, train_end],
            "valid": [train_end, valid_end],
            "test": [valid_end, test_end],
        },
        "files": {
            "z_series": "z_series.npy",
            "z_train": "z_train.npy",
            "z_valid": "z_valid.npy",
            "z_test": "z_test.npy",
            "u_reconstructed": "u_reconstructed.npy",
            "reconstruction_summary": "reconstruction_summary.csv",
            "reconstruction_mse_per_timestep": "reconstruction_mse_per_timestep.npy",
            "reconstruction_mae_per_timestep": "reconstruction_mae_per_timestep.npy",
        },
        "downstream_usage_note": (
            "Train RC/NGRC/PINN/PERC on z_train/z_valid/z_test. "
            "Decode predicted latent states using the decoder from the same run_dir."
        ),
    }

    (output_dir / "source_model_info.json").write_text(json.dumps(metadata, indent=2))

    notes = f"""# Final reproduced latent dataset: {args.label}

This folder contains latent data generated from a reproduced AE-SINDy run.

Source run:
`{run_dir}`

Source config:
`{args.source_config}`

Main files:

- `z_series.npy`: full latent trajectory
- `z_train.npy`: training latent data
- `z_valid.npy`: validation latent data
- `z_test.npy`: test latent data
- `u_reconstructed.npy`: decoded reconstruction in physical 64D space
- `reconstruction_summary.csv`: reconstruction statistics
- `source_model_info.json`: metadata for reproducibility

Use this latent dataset for downstream RC / NGRC / PINN / PERC experiments.
"""
    (output_dir / "README.md").write_text(notes)

    logger.info("saved latent data to: %s", output_dir)
    logger.info("z_series: %s", tuple(z_series.shape))
    logger.info("u_reconstructed: %s", tuple(u_reconstructed.shape))
    logger.info("summary: %s", summary_path)


if __name__ == "__main__":
    main()
