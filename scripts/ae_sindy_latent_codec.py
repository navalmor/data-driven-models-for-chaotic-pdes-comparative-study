from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from dataclasses import dataclass
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

logger = get_logger("ae_sindy.latent_codec")


@dataclass
class CodecInfo:
    run_dir: Path
    model_path: Path
    params_path: Path
    normalization_path: Path
    input_dim: int
    latent_dim: int
    normalize: bool
    device: str


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


def load_normalization(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    path = run_dir / "normalization_stats.json"

    if not path.exists():
        return np.asarray(0.0, dtype=np.float32), np.asarray(1.0, dtype=np.float32)

    stats = json.loads(path.read_text())
    mean = stats.get("mean", stats.get("data_mean", 0.0))
    std = stats.get("std", stats.get("data_std", 1.0))

    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


def iter_named_tensors(obj: Any, prefix: str = "") -> Iterable[Tuple[str, torch.Tensor]]:
    if torch.is_tensor(obj):
        yield prefix, obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_named_tensors(value, name)
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            name = f"{prefix}.{i}" if prefix else str(i)
            yield from iter_named_tensors(value, name)


def choose_tensor_by_keys(
    tensors: Dict[str, torch.Tensor],
    preferred_keys: list[str],
    batch_size: int,
    width: int,
) -> torch.Tensor | None:
    lower_map = {name.lower(): name for name in tensors.keys()}

    for wanted in preferred_keys:
        wanted_lower = wanted.lower()
        for existing_lower, original_name in lower_map.items():
            if existing_lower == wanted_lower or existing_lower.endswith("." + wanted_lower):
                tensor = tensors[original_name]
                if tensor.ndim == 2 and tensor.shape[0] == batch_size and tensor.shape[1] == width:
                    return tensor

    for wanted in preferred_keys:
        wanted_lower = wanted.lower()
        for existing_lower, original_name in lower_map.items():
            if wanted_lower in existing_lower:
                tensor = tensors[original_name]
                if tensor.ndim == 2 and tensor.shape[0] == batch_size and tensor.shape[1] == width:
                    return tensor

    return None


def select_tensor_by_shape(
    obj: Any,
    batch_size: int,
    width: int,
    label: str,
) -> torch.Tensor:
    if torch.is_tensor(obj):
        if obj.ndim == 2 and obj.shape[0] == batch_size and obj.shape[1] == width:
            return obj
        raise RuntimeError(
            f"{label} returned tensor with shape {tuple(obj.shape)}, "
            f"expected ({batch_size}, {width})."
        )

    tensors = {
        name: tensor
        for name, tensor in iter_named_tensors(obj)
        if tensor.ndim == 2 and tensor.shape[0] == batch_size and tensor.shape[1] == width
    }

    if tensors:
        return list(tensors.values())[0]

    raise RuntimeError(
        f"Could not select {label} tensor with expected shape ({batch_size}, {width})."
    )


def find_named_module_by_keywords(
    model: torch.nn.Module,
    include_keywords: list[str],
    exclude_keywords: list[str] | None = None,
) -> torch.nn.Module | None:
    """Find first submodule whose name contains all include keywords and none of the exclude keywords."""
    exclude_keywords = exclude_keywords or []

    candidates = []
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, torch.nn.ParameterList):
            continue
        low = name.lower()
        if all(k.lower() in low for k in include_keywords) and not any(k.lower() in low for k in exclude_keywords):
            candidates.append((name, module))

    # Prefer deeper named modules over the top-level container if several match.
    if candidates:
        candidates = sorted(candidates, key=lambda x: (x[0].count("."), len(x[0])), reverse=True)
        return candidates[0][1]

    return None


def call_module_safely(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Call a module and return its tensor output, allowing tuple/dict outputs."""
    out = module(x)
    return out



def apply_activation(x: torch.Tensor, activation_name: str) -> torch.Tensor:
    name = str(activation_name).lower()

    if name in {"elu", "torch.nn.elu"}:
        return torch.nn.functional.elu(x)
    if name in {"relu", "torch.nn.relu"}:
        return torch.relu(x)
    if name in {"tanh"}:
        return torch.tanh(x)
    if name in {"sigmoid"}:
        return torch.sigmoid(x)
    if name in {"gelu"}:
        return torch.nn.functional.gelu(x)
    if name in {"silu", "swish"}:
        return torch.nn.functional.silu(x)
    if name in {"softplus"}:
        return torch.nn.functional.softplus(x)
    if name in {"identity", "linear", "none"}:
        return x

    # ELU was the default activation used in the AE-SINDy experiments.
    return torch.nn.functional.elu(x)


def linear_from_weight_bias(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Apply one dense layer robustly.

    Some custom models store weights as (in_dim, out_dim), others as (out_dim, in_dim).
    This function detects the correct orientation from x.shape.
    """
    if x.shape[1] == weight.shape[0]:
        y = x @ weight
    elif x.shape[1] == weight.shape[1]:
        y = x @ weight.T
    else:
        raise RuntimeError(
            f"Cannot apply weight with shape {tuple(weight.shape)} "
            f"to input with shape {tuple(x.shape)}."
        )

    if bias.ndim == 1:
        if bias.shape[0] != y.shape[1]:
            raise RuntimeError(
                f"Bias shape {tuple(bias.shape)} does not match layer output {tuple(y.shape)}."
            )
        y = y + bias
    elif bias.ndim == 2:
        if bias.shape[-1] != y.shape[1]:
            raise RuntimeError(
                f"Bias shape {tuple(bias.shape)} does not match layer output {tuple(y.shape)}."
            )
        y = y + bias.reshape(1, -1)
    else:
        raise RuntimeError(f"Unsupported bias shape: {tuple(bias.shape)}")

    return y


def forward_parameter_mlp(
    x: torch.Tensor,
    weights: torch.nn.ParameterList,
    biases: torch.nn.ParameterList,
    activation_name: str,
    expected_width: int,
    label: str,
) -> torch.Tensor:
    """
    Forward pass through AE-SINDy networks stored as ParameterList objects.

    Hidden layers use the configured activation.
    Final layer is linear.
    """
    if len(weights) != len(biases):
        raise RuntimeError(
            f"{label}: number of weights and biases differ: {len(weights)} vs {len(biases)}."
        )

    h = x

    for i, (w, b) in enumerate(zip(weights, biases)):
        h = linear_from_weight_bias(h, w, b)

        if i < len(weights) - 1:
            h = apply_activation(h, activation_name)

    if h.ndim != 2 or h.shape[1] != expected_width:
        raise RuntimeError(
            f"{label}: expected output width {expected_width}, got shape {tuple(h.shape)}."
        )

    return h



class AESindyLatentCodec:
    """
    Reusable encoder/decoder bridge for trained AE-SINDy models.

    Main use cases:
      - encode physical 64D KSE data u(t) -> latent z(t)
      - decode latent predictions z_hat(t) -> physical 64D prediction u_hat(t)
      - export train/validation/test latent splits for RC / NGRC / PINN / PERC

    Important:
      - encoding input is assumed to be in original physical scale
      - decoding output is returned in original physical scale
      - normalization/unnormalization is handled internally using normalization_stats.json
    """

    def __init__(self, run_dir: str | Path, device: str = "auto"):
        self.run_dir = Path(run_dir)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.params = load_params(self.run_dir)

        if isinstance(self.params, dict):
            self.params = dict(self.params)
            self.params["device"] = torch.device(self.device)

        self.input_dim = int(get_param(self.params, "input_dim", 64))
        self.latent_dim = int(get_param(self.params, "latent_dim"))
        self.normalize = bool(get_param(self.params, "normalize", True))

        self.mean, self.std = load_normalization(self.run_dir)

        self.model = SINDyAutoencoderModel(self.params).to(self.device)
        self.model.load_state_dict(load_state_dict(self.run_dir))
        self.model.eval()

    @property
    def info(self) -> CodecInfo:
        return CodecInfo(
            run_dir=self.run_dir,
            model_path=self.run_dir / "model.pt",
            params_path=self.run_dir / "params.pkl",
            normalization_path=self.run_dir / "normalization_stats.json",
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            normalize=self.normalize,
            device=self.device,
        )

    def _normalize_u(self, u: np.ndarray) -> np.ndarray:
        if self.normalize:
            return ((u - self.mean) / self.std).astype(np.float32)
        return u.astype(np.float32)

    def _unnormalize_u(self, u_model: np.ndarray) -> np.ndarray:
        if self.normalize:
            return (u_model * self.std + self.mean).astype(np.float32)
        return u_model.astype(np.float32)

    def _validate_physical_input(self, u: np.ndarray) -> None:
        if u.ndim != 2:
            raise ValueError(f"Physical input must be 2D with shape (T, {self.input_dim}); got {u.shape}.")
        if u.shape[1] != self.input_dim:
            raise ValueError(
                f"Physical input has wrong dimension. Expected {self.input_dim}, got {u.shape[1]}."
            )

    def _validate_latent_input(self, z: np.ndarray) -> None:
        if z.ndim != 2:
            raise ValueError(f"Latent input must be 2D with shape (T, {self.latent_dim}); got {z.shape}.")
        if z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Latent input has wrong dimension. Expected {self.latent_dim}, got {z.shape[1]}. "
                f"This usually means you are using the wrong decoder for this latent data."
            )

    def _encode_batch_tensor(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # AE-SINDy project model stores encoder as ParameterLists.
        if hasattr(self.model, "encoder_weights") and hasattr(self.model, "encoder_biases"):
            activation_name = get_param(self.params, "activation", get_param(self.params, "activation_function", "elu"))
            return forward_parameter_mlp(
                x=x,
                weights=self.model.encoder_weights,
                biases=self.model.encoder_biases,
                activation_name=activation_name,
                expected_width=self.latent_dim,
                label="encoder_parameter_mlp",
            )

        # Direct method/module names.
        for name in ["encode", "encoder", "encoder_net", "encoder_network", "encoder_model"]:
            if hasattr(self.model, name):
                module = getattr(self.model, name)
                try:
                    out = module(x)
                    return select_tensor_by_shape(
                        out,
                        batch_size=batch_size,
                        width=self.latent_dim,
                        label=name,
                    )
                except Exception:
                    pass

        # Recursive named-module fallback.
        encoder_module = (
            find_named_module_by_keywords(self.model, ["encoder"])
            or find_named_module_by_keywords(self.model, ["enc"])
        )
        if encoder_module is not None:
            try:
                out = encoder_module(x)
                return select_tensor_by_shape(
                    out,
                    batch_size=batch_size,
                    width=self.latent_dim,
                    label="recursive_encoder_module",
                )
            except Exception:
                pass

        # Final fallback: full AE-SINDy forward and extract latent tensor.
        dx = torch.zeros_like(x)

        try:
            out = self.model(x, dx)
        except TypeError:
            out = self.model(x)

        tensors = {name: tensor for name, tensor in iter_named_tensors(out)}
        z = choose_tensor_by_keys(
            tensors,
            preferred_keys=["z", "z_latent", "latent", "encoded", "z0", "z_t"],
            batch_size=batch_size,
            width=self.latent_dim,
        )

        if z is not None:
            return z

        candidates = [
            (name, tensor)
            for name, tensor in tensors.items()
            if tensor.ndim == 2 and tensor.shape[0] == batch_size and tensor.shape[1] == self.latent_dim
        ]

        if candidates:
            return candidates[0][1]

        logger.debug("Available model output tensors:")
        for name, tensor in tensors.items():
            logger.debug("  %s: shape=%s", name, tuple(tensor.shape))

        raise RuntimeError(f"Could not extract latent tensor with shape ({batch_size}, {self.latent_dim}).")


    def _decode_batch_tensor(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]

        # AE-SINDy project model stores decoder as ParameterLists.
        if hasattr(self.model, "decoder_weights") and hasattr(self.model, "decoder_biases"):
            activation_name = get_param(self.params, "activation", get_param(self.params, "activation_function", "elu"))
            return forward_parameter_mlp(
                x=z,
                weights=self.model.decoder_weights,
                biases=self.model.decoder_biases,
                activation_name=activation_name,
                expected_width=self.input_dim,
                label="decoder_parameter_mlp",
            )

        # Direct method/module names.
        for name in ["decode", "decoder", "decoder_net", "decoder_network", "decoder_model"]:
            if hasattr(self.model, name):
                module = getattr(self.model, name)
                out = module(z)
                return select_tensor_by_shape(
                    out,
                    batch_size=batch_size,
                    width=self.input_dim,
                    label=name,
                )

        # Recursive named-module fallback.
        decoder_module = (
            find_named_module_by_keywords(self.model, ["decoder"])
            or find_named_module_by_keywords(self.model, ["dec"])
        )
        if decoder_module is not None:
            out = decoder_module(z)
            return select_tensor_by_shape(
                out,
                batch_size=batch_size,
                width=self.input_dim,
                label="recursive_decoder_module",
            )

        logger.debug("Available model modules:")
        for name, module in self.model.named_modules():
            if name:
                logger.debug("  %s: %s", name, type(module))

        raise RuntimeError(
            "Could not find decoder/decode module in model. "
            "Please inspect ae_sindy.model.SINDyAutoencoderModel for the decoder attribute name."
        )


    def encode(self, u_physical: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        u_physical = np.asarray(u_physical, dtype=np.float32)
        self._validate_physical_input(u_physical)

        u_model = self._normalize_u(u_physical)
        batches = []

        with torch.no_grad():
            for start in range(0, len(u_model), batch_size):
                batch = torch.from_numpy(u_model[start:start + batch_size]).to(self.device)
                z = self._encode_batch_tensor(batch)
                batches.append(z.detach().cpu().numpy())

        return np.concatenate(batches, axis=0).astype(np.float32)

    def decode(self, z: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        z = np.asarray(z, dtype=np.float32)
        self._validate_latent_input(z)

        batches = []

        with torch.no_grad():
            for start in range(0, len(z), batch_size):
                batch = torch.from_numpy(z[start:start + batch_size]).to(self.device)
                u_model = self._decode_batch_tensor(batch)
                batches.append(u_model.detach().cpu().numpy())

        u_model = np.concatenate(batches, axis=0).astype(np.float32)
        return self._unnormalize_u(u_model)

    def roundtrip(self, u_physical: np.ndarray, batch_size: int = 2048) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z = self.encode(u_physical, batch_size=batch_size)
        u_reconstructed = self.decode(z, batch_size=batch_size)

        mse_per_timestep = np.mean((u_reconstructed - u_physical) ** 2, axis=1).astype(np.float32)
        mae_per_timestep = np.mean(np.abs(u_reconstructed - u_physical), axis=1).astype(np.float32)

        return z, u_reconstructed, mse_per_timestep, mae_per_timestep

    def metadata(self, source_data_path: str | None = None) -> dict:
        info = self.info
        return {
            "run_dir": str(info.run_dir),
            "model_path": str(info.model_path),
            "params_path": str(info.params_path),
            "normalization_path": str(info.normalization_path),
            "source_data_path": source_data_path,
            "input_dim": info.input_dim,
            "latent_dim": info.latent_dim,
            "normalize": info.normalize,
            "device": info.device,
        }


def save_reconstruction_summary(
    path: Path,
    label: str,
    u_physical: np.ndarray,
    z: np.ndarray,
    mse_per_timestep: np.ndarray,
    mae_per_timestep: np.ndarray,
) -> None:
    with path.open("w", newline="") as f:
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
            "label": label,
            "n_samples": len(u_physical),
            "input_dim": u_physical.shape[1],
            "latent_dim": z.shape[1],
            "mse_mean": float(np.mean(mse_per_timestep)),
            "mse_min": float(np.min(mse_per_timestep)),
            "mse_max": float(np.max(mse_per_timestep)),
            "mse_std": float(np.std(mse_per_timestep)),
            "mae_mean": float(np.mean(mae_per_timestep)),
            "mae_min": float(np.min(mae_per_timestep)),
            "mae_max": float(np.max(mae_per_timestep)),
            "mae_std": float(np.std(mae_per_timestep)),
        })


def save_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2))


def cmd_info(args: argparse.Namespace) -> None:
    codec = AESindyLatentCodec(args.run_dir, device=args.device)
    info = codec.info

    logger.info("AE-SINDy latent codec info")
    logger.info("run_dir:       %s", info.run_dir)
    logger.info("model_path:    %s", info.model_path)
    logger.info("params_path:   %s", info.params_path)
    logger.info("norm_path:     %s", info.normalization_path)
    logger.info("input_dim:     %s", info.input_dim)
    logger.info("latent_dim:    %s", info.latent_dim)
    logger.info("normalize:     %s", info.normalize)
    logger.info("device:        %s", info.device)


def cmd_encode(args: argparse.Namespace) -> None:
    codec = AESindyLatentCodec(args.run_dir, device=args.device)

    u = np.load(args.input).astype(np.float32)
    z = codec.encode(u, batch_size=args.batch_size)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, z)

    if args.metadata:
        save_metadata(
            Path(args.metadata),
            {
                "operation": "encode",
                "input": args.input,
                "output": args.output,
                "input_shape": list(u.shape),
                "output_shape": list(z.shape),
                **codec.metadata(source_data_path=args.input),
            },
        )

    logger.info("saved: %s", output)
    logger.info("Input data: shape=%s (physical)", tuple(u.shape))
    logger.info("Latent data: shape=%s", tuple(z.shape))


def cmd_decode(args: argparse.Namespace) -> None:
    codec = AESindyLatentCodec(args.run_dir, device=args.device)

    z = np.load(args.input).astype(np.float32)
    u = codec.decode(z, batch_size=args.batch_size)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, u)

    if args.metadata:
        save_metadata(
            Path(args.metadata),
            {
                "operation": "decode",
                "input": args.input,
                "output": args.output,
                "input_shape": list(z.shape),
                "output_shape": list(u.shape),
                **codec.metadata(source_data_path=None),
            },
        )

    logger.info("saved: %s", output)
    logger.info("Latent data: shape=%s", tuple(z.shape))
    logger.info("Physical data: shape=%s", tuple(u.shape))


def cmd_roundtrip(args: argparse.Namespace) -> None:
    codec = AESindyLatentCodec(args.run_dir, device=args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    u = np.load(args.input).astype(np.float32)
    z, u_reconstructed, mse, mae = codec.roundtrip(u, batch_size=args.batch_size)

    np.save(output_dir / "z_series.npy", z)
    np.save(output_dir / "u_reconstructed.npy", u_reconstructed)
    np.save(output_dir / "reconstruction_mse_per_timestep.npy", mse)
    np.save(output_dir / "reconstruction_mae_per_timestep.npy", mae)

    save_reconstruction_summary(
        output_dir / "reconstruction_summary.csv",
        label=args.label,
        u_physical=u,
        z=z,
        mse_per_timestep=mse,
        mae_per_timestep=mae,
    )

    save_metadata(
        output_dir / "source_model_info.json",
        {
            "operation": "roundtrip",
            "input": args.input,
            "output_dir": str(output_dir),
            "input_shape": list(u.shape),
            "latent_shape": list(z.shape),
            "reconstructed_shape": list(u_reconstructed.shape),
            **codec.metadata(source_data_path=args.input),
        },
    )

    (output_dir / "README.md").write_text(
        f"""# AE-SINDy codec roundtrip: {args.label}

This folder was created by `scripts/ae_sindy_latent_codec.py roundtrip`.

Files:
- `z_series.npy`: encoded latent trajectory
- `u_reconstructed.npy`: decoded physical reconstruction
- `reconstruction_summary.csv`: reconstruction statistics
- `reconstruction_mse_per_timestep.npy`: timestep-wise reconstruction MSE
- `reconstruction_mae_per_timestep.npy`: timestep-wise reconstruction MAE
- `source_model_info.json`: metadata
"""
    )

    logger.info("saved roundtrip output: %s", output_dir)
    logger.info("Latent data: shape=%s", tuple(z.shape))
    logger.info("Reconstruction: shape=%s", tuple(u_reconstructed.shape))


def cmd_export_splits(args: argparse.Namespace) -> None:
    codec = AESindyLatentCodec(args.run_dir, device=args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    u = np.load(args.input).astype(np.float32)
    z, u_reconstructed, mse, mae = codec.roundtrip(u, batch_size=args.batch_size)

    train_end = args.train_end
    valid_end = args.valid_end
    test_end = len(z)

    if not (0 < train_end < valid_end <= test_end):
        raise ValueError(
            f"Invalid split boundaries: train_end={train_end}, valid_end={valid_end}, total={test_end}."
        )

    np.save(output_dir / "z_series.npy", z)
    np.save(output_dir / "z_train.npy", z[:train_end])
    np.save(output_dir / "z_valid.npy", z[train_end:valid_end])
    np.save(output_dir / "z_test.npy", z[valid_end:test_end])

    np.save(output_dir / "u_reconstructed.npy", u_reconstructed)
    np.save(output_dir / "reconstruction_mse_per_timestep.npy", mse)
    np.save(output_dir / "reconstruction_mae_per_timestep.npy", mae)

    save_reconstruction_summary(
        output_dir / "reconstruction_summary.csv",
        label=args.label,
        u_physical=u,
        z=z,
        mse_per_timestep=mse,
        mae_per_timestep=mae,
    )

    save_metadata(
        output_dir / "source_model_info.json",
        {
            "operation": "export-splits",
            "input": args.input,
            "output_dir": str(output_dir),
            "input_shape": list(u.shape),
            "latent_shape": list(z.shape),
            "reconstructed_shape": list(u_reconstructed.shape),
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
            },
            "downstream_usage_note": (
                "Train RC/NGRC/PINN/PERC on z_train/z_valid/z_test. "
                "Decode predicted latent states with this same run_dir."
            ),
            **codec.metadata(source_data_path=args.input),
        },
    )

    (output_dir / "README.md").write_text(
        f"""# Downstream latent dataset: {args.label}

Created by `scripts/ae_sindy_latent_codec.py export-splits`.

Use:
- `z_train.npy` for downstream training
- `z_valid.npy` for validation/model selection
- `z_test.npy` for final testing
- decode predicted latent trajectories with the same AE decoder from the recorded `run_dir`

Files:
- `z_series.npy`
- `z_train.npy`
- `z_valid.npy`
- `z_test.npy`
- `u_reconstructed.npy`
- `reconstruction_summary.csv`
- `source_model_info.json`
"""
    )

    logger.info("saved split export: %s", output_dir)
    logger.info("z_series: %s", tuple(z.shape))
    logger.info("z_train:  %s", tuple(z[:train_end].shape))
    logger.info("z_valid:  %s", tuple(z[train_end:valid_end].shape))
    logger.info("z_test:   %s", tuple(z[valid_end:test_end].shape))
    logger.info("u_reconstructed: %s", tuple(u_reconstructed.shape))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Production utility for encoding/decoding data with trained AE-SINDy models."
    )

    add_logging_arguments(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_run = argparse.ArgumentParser(add_help=False)
    common_run.add_argument("--run-dir", required=True, help="AE-SINDy run folder containing model.pt, params.pkl, normalization_stats.json.")
    common_run.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    common_run.add_argument("--batch-size", type=int, default=2048)

    p_info = subparsers.add_parser("info", parents=[common_run], help="Show codec/model information.")
    p_info.set_defaults(func=cmd_info)

    p_encode = subparsers.add_parser("encode", parents=[common_run], help="Encode physical data u(t) to latent z(t).")
    p_encode.add_argument("--input", required=True, help="Input physical .npy file with shape (T, input_dim).")
    p_encode.add_argument("--output", required=True, help="Output latent .npy file.")
    p_encode.add_argument("--metadata", default=None, help="Optional metadata JSON output path.")
    p_encode.set_defaults(func=cmd_encode)

    p_decode = subparsers.add_parser("decode", parents=[common_run], help="Decode latent data z(t) to physical u(t).")
    p_decode.add_argument("--input", required=True, help="Input latent .npy file with shape (T, latent_dim).")
    p_decode.add_argument("--output", required=True, help="Output physical .npy file.")
    p_decode.add_argument("--metadata", default=None, help="Optional metadata JSON output path.")
    p_decode.set_defaults(func=cmd_decode)

    p_roundtrip = subparsers.add_parser("roundtrip", parents=[common_run], help="Encode then decode physical data and save reconstruction diagnostics.")
    p_roundtrip.add_argument("--input", required=True, help="Input physical .npy file with shape (T, input_dim).")
    p_roundtrip.add_argument("--output-dir", required=True)
    p_roundtrip.add_argument("--label", default="ae_sindy_roundtrip")
    p_roundtrip.set_defaults(func=cmd_roundtrip)

    p_export = subparsers.add_parser("export-splits", parents=[common_run], help="Export z_series/z_train/z_valid/z_test and diagnostics.")
    p_export.add_argument("--input", required=True, help="Input physical .npy file with shape (T, input_dim).")
    p_export.add_argument("--output-dir", required=True)
    p_export.add_argument("--label", default="ae_sindy_latent_export")
    p_export.add_argument("--train-end", type=int, default=20000)
    p_export.add_argument("--valid-end", type=int, default=25000)
    p_export.set_defaults(func=cmd_export_splits)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_from_args(args, component="AE-SINDy")
    args.func(args)


if __name__ == "__main__":
    main()
