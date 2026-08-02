from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import logging
import pickle
import math
from pathlib import Path
import random
import shutil
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from forecasting.data import complete_trajectory_shape
from forecasting.models.pinn import StatePredictor
from forecasting.progress import progress_indices

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PINNRunResult:
    output_dir: Path
    summary_path: Path
    config_resolved_path: Path


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False

    # PYTHONHASHSEED and CUBLAS_WORKSPACE_CONFIG must be exported before the
    # interpreter starts, so they cannot be set here; scripts/repro_env.sh does it.
    # warn_only=True because a few ops have no deterministic kernel in torch 2.6
    # and would otherwise raise mid-training.
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except Exception as exc:
        logger.warning("Could not enable deterministic algorithms: %s", exc)


def _train_standardizer(train: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    # Component-wise train-only standardization, shape (nx,).
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _standardize(u: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((u - mean) / std).astype(np.float32)


def _unstandardize_torch(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return z * std + mean


class KSESpectralResidual:
    def __init__(
        self,
        nx: int,
        domain_length: float,
        dt: float,
        *,
        dealias: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.nx = int(nx)
        self.domain_length = float(domain_length)
        self.dt = float(dt)
        self.dealias = bool(dealias)
        k = 2.0 * math.pi * torch.fft.fftfreq(
            self.nx,
            d=self.domain_length / self.nx,
            device=device,
        ).to(dtype=dtype)
        self.k = k
        self.l_hat = k.pow(2) - k.pow(4)
        if self.dealias:
            kmax = torch.max(torch.abs(k))
            cutoff = (2.0 / 3.0) * kmax
            self.dealias_mask = (torch.abs(k) <= cutoff).to(device=device)
        else:
            self.dealias_mask = torch.ones_like(k, dtype=torch.bool, device=device)

    def nonlinear_hat(self, u: torch.Tensor) -> torch.Tensor:
        u2_hat = torch.fft.fft(u * u, dim=-1)
        n_hat = -0.5j * self.k * u2_hat
        if self.dealias:
            n_hat = n_hat * self.dealias_mask
        return n_hat

    def cnab2_residual(
        self,
        u_prev: torch.Tensor,
        u_cur: torch.Tensor,
        u_next_pred: torch.Tensor,
    ) -> torch.Tensor:
        u_next_hat = torch.fft.fft(u_next_pred, dim=-1)
        u_cur_hat = torch.fft.fft(u_cur, dim=-1)
        n_cur = self.nonlinear_hat(u_cur)
        n_prev = self.nonlinear_hat(u_prev)

        residual_hat = (
            (1.0 - 0.5 * self.dt * self.l_hat) * u_next_hat
            - (1.0 + 0.5 * self.dt * self.l_hat) * u_cur_hat
            - self.dt * (1.5 * n_cur - 0.5 * n_prev)
        )
        return torch.fft.ifft(residual_hat, dim=-1).real

    def backward_euler_residual(
        self,
        u_cur: torch.Tensor,
        u_next_pred: torch.Tensor,
    ) -> torch.Tensor:
        # R = (u_{n+1}-u_n)/dt + 0.5 d_x(u_{n+1}^2) + u_xx + u_xxxx.
        u_hat = torch.fft.fft(u_next_pred, dim=-1)
        nonlinear = 0.5 * torch.fft.ifft(1j * self.k * torch.fft.fft(u_next_pred * u_next_pred, dim=-1), dim=-1).real
        u_xx = torch.fft.ifft(-(self.k.pow(2)) * u_hat, dim=-1).real
        u_xxxx = torch.fft.ifft(self.k.pow(4) * u_hat, dim=-1).real
        return (u_next_pred - u_cur) / self.dt + nonlinear + u_xx + u_xxxx



class FrozenAEDecoder(nn.Module):
    """
    Differentiable frozen AE-SINDy decoder for latent PINN physics.

    It maps latent coordinates z in AE latent scale to physical KSE states u
    in original physical scale. Gradients flow through z into the PINN
    predictor, while decoder weights remain frozen.
    """

    def __init__(self, run_dir: Path, *, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.run_dir = Path(run_dir)

        params_path = self.run_dir / "params.pkl"
        norm_path = self.run_dir / "normalization_stats.json"
        ckpt_path = self.run_dir / "model.pt"

        if not params_path.exists():
            raise FileNotFoundError(f"Missing AE params.pkl: {params_path}")
        if not norm_path.exists():
            raise FileNotFoundError(f"Missing AE normalization_stats.json: {norm_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing AE model.pt: {ckpt_path}")

        with params_path.open("rb") as f:
            params = pickle.load(f)

        with norm_path.open("r") as f:
            norm = json.load(f)

        state = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(state, dict):
            raise TypeError(f"Expected AE checkpoint state_dict, got {type(state)}")

        self.input_dim = int(params.get("input_dim", 64))
        self.latent_dim = int(params["latent_dim"])
        self.activation_name = str(params.get("activation", "elu")).lower()
        self.normalize = bool(params.get("normalize", True))

        mean = torch.as_tensor(norm.get("mean", 0.0), dtype=dtype, device=device)
        std = torch.as_tensor(norm.get("std", 1.0), dtype=dtype, device=device)
        self.register_buffer("physical_mean", mean)
        self.register_buffer("physical_std", std)

        weight_keys = sorted(
            [k for k in state if k.startswith("decoder_weights.")],
            key=lambda k: int(k.split(".")[-1]),
        )
        bias_keys = sorted(
            [k for k in state if k.startswith("decoder_biases.")],
            key=lambda k: int(k.split(".")[-1]),
        )
        if not weight_keys or len(weight_keys) != len(bias_keys):
            raise RuntimeError(
                f"Invalid decoder checkpoint: {len(weight_keys)} weights, {len(bias_keys)} biases."
            )

        self.num_layers = len(weight_keys)
        for i, key in enumerate(weight_keys):
            self.register_buffer(f"decoder_weight_{i}", state[key].to(device=device, dtype=dtype))
        for i, key in enumerate(bias_keys):
            self.register_buffer(f"decoder_bias_{i}", state[key].to(device=device, dtype=dtype))

        first_w = getattr(self, "decoder_weight_0")
        last_w = getattr(self, f"decoder_weight_{self.num_layers - 1}")
        if first_w.shape[0] != self.latent_dim:
            raise RuntimeError(
                f"Decoder latent input mismatch: params latent_dim={self.latent_dim}, "
                f"checkpoint first weight shape={tuple(first_w.shape)}."
            )
        if last_w.shape[1] != self.input_dim:
            raise RuntimeError(
                f"Decoder physical output mismatch: params input_dim={self.input_dim}, "
                f"checkpoint last weight shape={tuple(last_w.shape)}."
            )

        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "elu":
            return torch.nn.functional.elu(x)
        if self.activation_name == "relu":
            return torch.relu(x)
        if self.activation_name == "sigmoid":
            return torch.sigmoid(x)
        if self.activation_name == "tanh":
            return torch.tanh(x)
        if self.activation_name == "linear":
            return x
        raise ValueError(f"Unsupported AE decoder activation: {self.activation_name}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent tensor shape (batch, {self.latent_dim}), got {tuple(z.shape)}"
            )

        h = z
        for i in range(self.num_layers):
            w = getattr(self, f"decoder_weight_{i}")
            b = getattr(self, f"decoder_bias_{i}")
            h = torch.matmul(h, w) + b
            if i < self.num_layers - 1:
                h = self._activation(h)

        if h.shape[1] != self.input_dim:
            raise RuntimeError(f"Decoded output shape mismatch: {tuple(h.shape)}")

        if self.normalize:
            h = h * self.physical_std + self.physical_mean
        return h


def _resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _load_2d_array(path: Path, *, name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    arr = np.load(path, allow_pickle=False).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape} at {path}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf: {path}")
    return arr


def _decode_native_array(
    decoder: FrozenAEDecoder,
    native: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    decoder.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(native), batch_size):
            z = torch.as_tensor(native[start:start + batch_size], device=device, dtype=torch.float32)
            u = decoder(z)
            out.append(u.detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def _compute_physics_scale(train: np.ndarray, mode: str = "delta_rms") -> float:
    if mode == "delta_rms":
        delta = train[1:] - train[:-1]
        scale = float(np.sqrt(np.mean(delta * delta)))
    elif mode == "state_rms":
        scale = float(np.sqrt(np.mean(train * train)))
    else:
        raise ValueError(f"Unsupported physics_scale mode: {mode}")
    return max(scale, 1e-8)


def _relative_l2_metrics(
    truth: np.ndarray,
    pred: np.ndarray,
    *,
    dt: float,
    threshold: float,
    safety_max_abs: float,
    split_dir: Path,
) -> dict[str, Any]:
    split_dir.mkdir(parents=True, exist_ok=True)

    denom = np.linalg.norm(truth, axis=1)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    rel = np.linalg.norm(pred - truth, axis=1) / denom
    max_abs = np.max(np.abs(pred), axis=1)
    finite = np.isfinite(rel) & np.isfinite(max_abs) & (max_abs <= safety_max_abs)
    safe = finite & (rel <= threshold)

    first_unsafe_index: int | None = None
    for i, ok in enumerate(safe):
        if not bool(ok):
            first_unsafe_index = int(i)
            break

    safe_prefix_length = int(len(safe) if first_unsafe_index is None else first_unsafe_index)
    horizon_time = float(safe_prefix_length * dt)

    finite_rel = rel[np.isfinite(rel)]
    metrics = {
        "relative_l2_horizon_time": horizon_time,
        "relative_l2_mean": float(np.mean(finite_rel)) if finite_rel.size else float("nan"),
        "relative_l2_final": float(rel[-1]) if len(rel) else float("nan"),
        "finite_relative_l2_fraction": float(np.mean(finite)),
        "first_unsafe_index": first_unsafe_index,
        "safe_prefix_length": safe_prefix_length,
        "relative_l2_threshold": float(threshold),
        "prediction_safety_max_abs": float(safety_max_abs),
        "num_steps": int(len(rel)),
    }

    with (split_dir / "errors_physical.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "time",
                "relative_l2",
                "prediction_max_abs",
                "finite",
                "safe",
            ],
        )
        writer.writeheader()
        for i, value in enumerate(rel):
            writer.writerow(
                {
                    "index": i,
                    "time": i * dt,
                    "relative_l2": value,
                    "prediction_max_abs": max_abs[i],
                    "finite": bool(finite[i]),
                    "safe": bool(safe[i]),
                }
            )

    (split_dir / "metrics_physical.json").write_text(
        json.dumps(_to_jsonable(metrics), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    np.save(split_dir / "truth_physical.npy", truth)
    np.save(split_dir / "pred_physical.npy", pred)

    return metrics


def _rollout(
    model: nn.Module,
    truth: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    z_truth = _standardize(truth, mean, std)
    pred_z = np.empty_like(z_truth, dtype=np.float32)
    pred_z[0] = z_truth[0]

    z = torch.as_tensor(z_truth[0:1], device=device, dtype=torch.float32)
    with torch.no_grad():
        for i in range(1, len(z_truth)):
            z = model(z)
            pred_z[i] = z.detach().cpu().numpy()[0]

    pred = pred_z * std[None, :] + mean[None, :]
    return pred.astype(np.float32)


def _append_registry(
    *,
    repo_root: Path,
    config_path: Path | None,
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    registry_path = repo_root / "experiments" / "forecasting" / "summary_tables" / "forecasting_experiment_registry.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_id",
        "model_name",
        "run_type",
        "data_mode",
        "native_dimension",
        "physical_dimension",
        "train_end_index",
        "valid_end_index",
        "standardized",
        "decoded_to_physical",
        "selection_metric",
        "valid_relative_l2_horizon_time",
        "test_relative_l2_horizon_time",
        "valid_relative_l2_mean",
        "test_relative_l2_mean",
        "valid_relative_l2_final",
        "test_relative_l2_final",
        "config_path",
        "output_dir",
    ]

    valid = summary.get("validation_metrics_physical") or {}
    test = summary.get("test_metrics_physical") or {}

    row = {
        "run_id": summary["run_id"],
        "model_name": "pinn",
        "run_type": summary.get("run_type", "single"),
        "data_mode": summary.get("data_mode", "full64"),
        "native_dimension": summary.get("native_dimension", ""),
        "physical_dimension": summary.get("physical_dimension", ""),
        "train_end_index": summary.get("train_end_index", ""),
        "valid_end_index": summary.get("valid_end_index", ""),
        "standardized": str(bool(summary.get("standardized", True))).lower(),
        "decoded_to_physical": str(bool(summary.get("decoded_to_physical", False))).lower(),
        "selection_metric": "relative_l2_horizon",
        "valid_relative_l2_horizon_time": valid.get("relative_l2_horizon_time", ""),
        "test_relative_l2_horizon_time": test.get("relative_l2_horizon_time", ""),
        "valid_relative_l2_mean": valid.get("relative_l2_mean", ""),
        "test_relative_l2_mean": test.get("relative_l2_mean", ""),
        "valid_relative_l2_final": valid.get("relative_l2_final", ""),
        "test_relative_l2_final": test.get("relative_l2_final", ""),
        "config_path": str(config_path) if config_path is not None else "",
        "output_dir": str(output_dir.relative_to(repo_root)) if output_dir.is_absolute() else str(output_dir),
    }

    exists = registry_path.exists()
    with registry_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_pinn_experiment(
    cfg: dict[str, Any],
    *,
    repo_root: Path,
    config_path: Path | None = None,
    append_registry: bool | None = None,
    evaluate_test: bool | None = None,
) -> PINNRunResult:
    _t0 = time.time()
    experiment = cfg.get("experiment", {})
    data_cfg = cfg.get("data", {})
    prep_cfg = cfg.get("preprocessing", {})
    eval_cfg = cfg.get("evaluation", {})
    model_cfg = cfg.get("model", {})
    output_cfg = cfg.get("output", {})

    run_id = experiment.get("run_id")
    if not run_id:
        raise ValueError("experiment.run_id is required")

    # Allow search/probe scripts to override run_id while preserving the
    # original config template. This keeps candidate outputs distinct.
    run_id_override = cfg.get("_runtime", {}).get("run_id_override")
    if run_id_override:
        run_id = str(run_id_override)
        experiment = {**experiment, "run_id": run_id}
    if experiment.get("model") != "pinn":
        raise ValueError("experiment.model must be 'pinn'")
    data_mode = str(experiment.get("data_mode", "full64"))
    if data_mode not in {"full64", "latent8"}:
        raise ValueError("PINN runner supports data_mode='full64' or data_mode='latent8' only")

    logger.info("Starting PINN run: run_id=%s data_mode=%s", run_id, data_mode)

    append_registry_flag = bool(output_cfg.get("append_registry", False) if append_registry is None else append_registry)
    evaluate_test_flag = bool(output_cfg.get("evaluate_test", False) if evaluate_test is None else evaluate_test)

    base_dir = repo_root / output_cfg.get("base_dir", "experiments/forecasting")
    run_group = cfg.get("_runtime", {}).get("run_group_override", output_cfg.get("run_group", "smoke"))
    output_dir = base_dir / "pinn" / run_group / run_id
    overwrite = bool(output_cfg.get("overwrite", False))
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory exists and overwrite=false: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(model_cfg.get("random_seed", 42))
    _set_seed(seed)

    train_end = int(data_cfg.get("train_end_index", 20000))
    valid_end = int(data_cfg.get("valid_end_index", 25000))

    native_data_dir: Path | None = None
    decoder_run_dir: Path | None = None

    if data_mode == "full64":
        data_path = _resolve_repo_path(
            repo_root,
            data_cfg.get("data_path", "simulation/simulation_data/u_series.npy"),
        )
        physical_all = _load_2d_array(data_path, name="physical u_series")
        if not (0 < train_end < valid_end < len(physical_all)):
            raise ValueError(
                f"Invalid split train_end={train_end} valid_end={valid_end} len={len(physical_all)}"
            )

        train = physical_all[:train_end]
        valid = physical_all[train_end:valid_end]
        test = physical_all[valid_end:]

        physical_train = train
        physical_valid = valid
        physical_test = test
        summary_data_path = data_path
        decoded_to_physical = False

    else:
        physical_data_path = _resolve_repo_path(
            repo_root,
            data_cfg.get("physical_data_path", "simulation/simulation_data/u_series.npy"),
        )
        native_data_dir = _resolve_repo_path(repo_root, data_cfg["native_data_dir"])
        decoder_run_dir = _resolve_repo_path(repo_root, data_cfg["decoder_run_dir"])

        physical_all = _load_2d_array(physical_data_path, name="physical u_series")
        if not (0 < train_end < valid_end < len(physical_all)):
            raise ValueError(
                f"Invalid split train_end={train_end} valid_end={valid_end} len={len(physical_all)}"
            )

        z_train_path = native_data_dir / "z_train.npy"
        z_valid_path = native_data_dir / "z_valid.npy"
        z_test_path = native_data_dir / "z_test.npy"
        z_series_path = native_data_dir / "z_series.npy"

        if z_train_path.exists() and z_valid_path.exists() and z_test_path.exists():
            train = _load_2d_array(z_train_path, name="z_train")
            valid = _load_2d_array(z_valid_path, name="z_valid")
            test = _load_2d_array(z_test_path, name="z_test")
        else:
            z_all = _load_2d_array(z_series_path, name="z_series")
            train = z_all[:train_end]
            valid = z_all[train_end:valid_end]
            test = z_all[valid_end:]

        physical_train = physical_all[:train_end]
        physical_valid = physical_all[train_end:valid_end]
        physical_test = physical_all[valid_end:]

        if len(train) != len(physical_train) or len(valid) != len(physical_valid) or len(test) != len(physical_test):
            raise ValueError(
                "Latent/physical split length mismatch: "
                f"train {len(train)}/{len(physical_train)}, "
                f"valid {len(valid)}/{len(physical_valid)}, "
                f"test {len(test)}/{len(physical_test)}"
            )

        summary_data_path = physical_data_path
        decoded_to_physical = True

    standardize = bool(prep_cfg.get("standardize", True))
    if not standardize:
        raise ValueError("PINN runner currently requires preprocessing.standardize=true")

    mean, std = _train_standardizer(train)
    z_train = _standardize(train, mean, std)

    dt = float(eval_cfg.get("dt", 0.1))
    threshold = float(eval_cfg.get("relative_l2_threshold", 0.5))
    safety_max_abs = float(eval_cfg.get("prediction_safety_max_abs", 5.0))

    native_dim = int(model_cfg.get("native_dimension", train.shape[1]))
    if native_dim != train.shape[1]:
        raise ValueError(f"native_dimension={native_dim} does not match native data dimension {train.shape[1]}")

    physical_dim = int(model_cfg.get("physical_dimension", physical_train.shape[1]))
    if physical_dim != physical_train.shape[1]:
        raise ValueError(
            f"physical_dimension={physical_dim} does not match physical data dimension {physical_train.shape[1]}"
        )

    domain_length = float(model_cfg.get("domain_length", 22.0))

    representation = "latent space" if data_mode == "latent8" else "full state"
    logger.info("Input trajectory: shape=%s, representation=%s, dtype=%s",
                complete_trajectory_shape(train, valid, test), representation, train.dtype)
    logger.debug("Data splits: train=%s, validation=%s, test=%s",
                 tuple(train.shape), tuple(valid.shape), tuple(test.shape))
    logger.debug("split indices: train_end=%d valid_end=%d native_dim=%d physical_dim=%d",
                 train_end, valid_end, native_dim, physical_dim)

    device_name = str(model_cfg.get("device", "auto"))
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            # Automatic selection fell back to CPU: worth flagging, but CPU is a
            # fully supported (and for this package, the validated) execution path.
            logger.warning("CUDA unavailable; using CPU")
    else:
        device = torch.device(device_name)
    logger.info("Device: %s", device)

    decoder: FrozenAEDecoder | None = None
    if data_mode == "latent8":
        if decoder_run_dir is None:
            raise ValueError("data.decoder_run_dir is required for latent8 PINN")
        decoder = FrozenAEDecoder(decoder_run_dir, device=device, dtype=torch.float32).to(device)
        decoder.eval()

    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    depth = int(model_cfg.get("depth", 2))
    activation = str(model_cfg.get("activation", "tanh"))
    residual_prediction = bool(model_cfg.get("residual_prediction", True))

    model = StatePredictor(
        dim=native_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        activation=activation,
        residual_prediction=residual_prediction,
    ).to(device)

    batch_size = int(model_cfg.get("batch_size", 128))
    epochs = int(model_cfg.get("epochs", 5))
    batches_per_epoch = int(model_cfg.get("batches_per_epoch", 20))
    lr = float(model_cfg.get("learning_rate", 1e-3))
    weight_decay = float(model_cfg.get("weight_decay", 0.0))
    lambda_phy = float(model_cfg.get("lambda_phy", 0.0))
    beta_multistep = float(model_cfg.get("beta_multistep", 0.0))
    multistep_horizon = int(model_cfg.get("multistep_horizon", 1))
    physics_residual = str(model_cfg.get("physics_residual", "cnab2")).lower()
    dealias = bool(model_cfg.get("dealias", True))
    grad_clip = model_cfg.get("gradient_clip_norm", None)
    grad_clip_value = None if grad_clip is None else float(grad_clip)

    if multistep_horizon < 1:
        raise ValueError("multistep_horizon must be >= 1")

    z_train_t = torch.as_tensor(z_train, device=device, dtype=torch.float32)
    u_train_t = torch.as_tensor(physical_train, device=device, dtype=torch.float32)
    mean_t = torch.as_tensor(mean, device=device, dtype=torch.float32)
    std_t = torch.as_tensor(std, device=device, dtype=torch.float32)

    residual = KSESpectralResidual(
        nx=physical_dim,
        domain_length=domain_length,
        dt=dt,
        dealias=dealias,
        device=device,
        dtype=torch.float32,
    )
    physics_scale = _compute_physics_scale(physical_train, str(model_cfg.get("physics_scale", "delta_rms")))
    physics_scale_t = torch.tensor(physics_scale, device=device, dtype=torch.float32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    max_start = train_end - 1 - multistep_horizon
    if max_start <= 1:
        raise ValueError("Training split too short for requested multistep_horizon")

    logger.info("Training PINN: %d epochs x %d batches (batch_size=%d, lr=%g)",
                epochs, batches_per_epoch, batch_size, lr)
    # Deterministic reporting cadence: ~10 updates in the default view, always
    # including the first and last epoch (every epoch under --verbose/--debug).
    # This only decides whether to emit an already-computed value; it never
    # touches the optimiser, gradients, RNG or loss.
    _progress_epochs = progress_indices(epochs)

    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_total = 0.0
        epoch_data = 0.0
        epoch_ms = 0.0
        epoch_phy = 0.0

        for _ in range(batches_per_epoch):
            idx = torch.randint(1, max_start + 1, (batch_size,), device=device)

            z = z_train_t[idx]
            pred1 = model(z)
            target1 = z_train_t[idx + 1]
            loss_data = torch.mean((pred1 - target1) ** 2)

            loss_ms = torch.zeros((), device=device)
            z_roll = pred1
            if multistep_horizon > 1 and beta_multistep != 0.0:
                ms_terms = []
                for step in range(2, multistep_horizon + 1):
                    z_roll = model(z_roll)
                    target = z_train_t[idx + step]
                    ms_terms.append(torch.mean((z_roll - target) ** 2) / float(step))
                loss_ms = torch.stack(ms_terms).sum() if ms_terms else loss_ms

            loss_phy = torch.zeros((), device=device)
            if lambda_phy != 0.0:
                native_next_pred = _unstandardize_torch(pred1, mean_t, std_t)
                if decoder is None:
                    u_next_pred = native_next_pred
                else:
                    u_next_pred = decoder(native_next_pred)
                u_cur = u_train_t[idx]
                if physics_residual == "cnab2":
                    u_prev = u_train_t[idx - 1]
                    r = residual.cnab2_residual(u_prev, u_cur, u_next_pred)
                    loss_phy = torch.mean((r / physics_scale_t) ** 2)
                elif physics_residual == "backward_euler":
                    r = residual.backward_euler_residual(u_cur, u_next_pred)
                    loss_phy = torch.mean((r / (physics_scale_t / dt)) ** 2)
                else:
                    raise ValueError(f"Unsupported physics_residual: {physics_residual}")

            loss = loss_data + beta_multistep * loss_ms + lambda_phy * loss_phy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            optimizer.step()

            epoch_total += float(loss.detach().cpu())
            epoch_data += float(loss_data.detach().cpu())
            epoch_ms += float(loss_ms.detach().cpu())
            epoch_phy += float(loss_phy.detach().cpu())

        denom = float(batches_per_epoch)
        history.append(
            {
                "epoch": float(epoch),
                "loss_total": epoch_total / denom,
                "loss_data": epoch_data / denom,
                "loss_multistep": epoch_ms / denom,
                "loss_physics": epoch_phy / denom,
            }
        )
        if epoch in _progress_epochs:
            logger.info(
                "PROGRESS | Epoch %d/%d | loss=%.6g data=%.6g ms=%.6g phy=%.6g",
                epoch, epochs,
                history[-1]["loss_total"], history[-1]["loss_data"],
                history[-1]["loss_multistep"], history[-1]["loss_physics"],
            )

    # Save model and standardizer.
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {
                "dim": native_dim,
                "hidden_dim": hidden_dim,
                "depth": depth,
                "activation": activation,
                "residual_prediction": residual_prediction,
            },
            "mean": mean,
            "std": std,
        },
        output_dir / "pinn_model.pt",
    )
    np.savez(output_dir / "standardizer.npz", mean=mean, std=std)

    # Validation rollout in native space; metrics are always physical-space metrics.
    valid_pred_native = _rollout(model, valid, mean, std, device=device)
    if decoder is None:
        valid_pred_physical = valid_pred_native
    else:
        valid_pred_physical = _decode_native_array(decoder, valid_pred_native, device=device)
        np.save(output_dir / "validation_pred_native.npy", valid_pred_native)

    valid_metrics = _relative_l2_metrics(
        physical_valid,
        valid_pred_physical,
        dt=dt,
        threshold=threshold,
        safety_max_abs=safety_max_abs,
        split_dir=output_dir / "validation",
    )

    test_metrics: dict[str, Any] | None = None
    if evaluate_test_flag:
        test_pred_native = _rollout(model, test, mean, std, device=device)
        if decoder is None:
            test_pred_physical = test_pred_native
        else:
            test_pred_physical = _decode_native_array(decoder, test_pred_native, device=device)
            np.save(output_dir / "test_pred_native.npy", test_pred_native)
        test_metrics = _relative_l2_metrics(
            physical_test,
            test_pred_physical,
            dt=dt,
            threshold=threshold,
            safety_max_abs=safety_max_abs,
            split_dir=output_dir / "test",
        )

    feature_dimension = int(sum(p.numel() for p in model.parameters()))
    logger.debug("trainable parameters: %d", feature_dimension)
    # relative_l2_horizon_time is horizon_steps * dt (see forecasting.metrics), i.e.
    # physical simulation time, matching validation_horizon_physical in the public
    # tables. The step count is the companion authoritative field; no new conversion.
    logger.info("METRIC | Validation horizon (physical time): %s",
                valid_metrics.get("relative_l2_horizon_time"))
    logger.info("METRIC | Validation relative-L2 mean: %s", valid_metrics.get("relative_l2_mean"))
    logger.debug("Validation horizon: %s steps", valid_metrics.get("relative_l2_horizon_steps"))
    if test_metrics is not None:
        logger.info("METRIC | Test horizon (physical time): %s",
                    test_metrics.get("relative_l2_horizon_time"))
        logger.info("METRIC | Test relative-L2 mean: %s", test_metrics.get("relative_l2_mean"))
        logger.debug("Test horizon: %s steps", test_metrics.get("relative_l2_horizon_steps"))

    resolved_cfg = _to_jsonable(cfg)
    (output_dir / "config_input.json").write_text(
        json.dumps(_to_jsonable(cfg), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    config_resolved_path = output_dir / "config_resolved.json"
    config_resolved_path.write_text(
        json.dumps(resolved_cfg, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    with (output_dir / "training_history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "loss_total", "loss_data", "loss_multistep", "loss_physics"],
        )
        writer.writeheader()
        writer.writerows(history)

    summary: dict[str, Any] = {
        "run_id": run_id,
        "model_name": "pinn",
        "model": "pinn",
        "run_type": experiment.get("type", "single"),
        "data_mode": data_mode,
        "native_dimension": native_dim,
        "physical_dimension": physical_dim,
        "decoded_to_physical": decoded_to_physical,
        "standardized": True,
        "train_end_index": train_end,
        "valid_end_index": valid_end,
        "evaluate_test": evaluate_test_flag,
        "append_registry": append_registry_flag,
        "feature_dimension": feature_dimension,
        "config_path": str(config_path) if config_path is not None else None,
        "output_dir": str(output_dir.relative_to(repo_root)),
        "data_path": str(summary_data_path.relative_to(repo_root)) if summary_data_path.is_absolute() else str(summary_data_path),
        "physical_data_path": str(summary_data_path.relative_to(repo_root)) if summary_data_path.is_absolute() else str(summary_data_path),
        "native_data_dir": (
            str(native_data_dir.relative_to(repo_root)) if native_data_dir is not None and native_data_dir.is_absolute()
            else (str(native_data_dir) if native_data_dir is not None else None)
        ),
        "decoder_run_dir": (
            str(decoder_run_dir.relative_to(repo_root)) if decoder_run_dir is not None and decoder_run_dir.is_absolute()
            else (str(decoder_run_dir) if decoder_run_dir is not None else None)
        ),
        "validation_metrics_physical": valid_metrics,
        "test_metrics_physical": test_metrics or {},
        "pinn": {
            "architecture": {
                "hidden_dim": hidden_dim,
                "depth": depth,
                "activation": activation,
                "residual_prediction": residual_prediction,
                "num_parameters": feature_dimension,
            },
            "training": {
                "epochs": epochs,
                "batches_per_epoch": batches_per_epoch,
                "batch_size": batch_size,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "multistep_horizon": multistep_horizon,
                "beta_multistep": beta_multistep,
                "lambda_phy": lambda_phy,
                "physics_residual": physics_residual,
                "physics_scale": physics_scale,
                "gradient_clip_norm": grad_clip_value,
                "random_seed": seed,
                "device": str(device),
            },
            "kse_physics": {
                "equation": "u_t = -0.5*d_x(u^2) - u_xx - u_xxxx",
                "domain_length": domain_length,
                "dt": dt,
                "nx": physical_dim,
                "dealias": dealias,
                "residual_space": "physical_full64" if decoder is None else "decoded_physical_full64",
                "latent_decoder_backprop": bool(decoder is not None),
                "spectral_wavenumber": "2*pi*fftfreq(nx, d=domain_length/nx)",
            },
        },
        "metadata": cfg.get("metadata", {}),
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_to_jsonable(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    if append_registry_flag:
        _append_registry(
            repo_root=repo_root,
            config_path=config_path,
            summary=summary,
            output_dir=output_dir,
        )

    logger.info("Output: %s", output_dir.relative_to(repo_root))
    logger.info("Completed in %.1f s", time.time() - _t0)

    return PINNRunResult(
        output_dir=output_dir,
        summary_path=summary_path,
        config_resolved_path=config_resolved_path,
    )
