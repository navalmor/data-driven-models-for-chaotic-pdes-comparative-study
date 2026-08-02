from __future__ import annotations

import csv
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from .configs import AESINDyConfig
from .logging_utils import get_logger
from .losses import define_loss_torch
from .model import SINDyAutoencoderModel

logger = get_logger("train")
ParamsLike = Union[Dict[str, Any], AESINDyConfig]

def resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)

def set_random_seed(seed: int, deterministic: bool = True) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(deterministic)
    except Exception as exc:
        logger.warning("Could not enable deterministic algorithms: %s", exc)



def coerce_params(params: ParamsLike, run_dir: Optional[Path] = None) -> Dict[str, Any]:
    if isinstance(params, AESINDyConfig):
        out = params.to_params_dict(run_dir=run_dir)
    else:
        out = dict(params)
        if "library_dim" not in out:
            raise ValueError("params must include library_dim")
        if "coefficient_mask" not in out:
            out["coefficient_mask"] = np.ones((out["library_dim"], out["latent_dim"]), dtype=np.float32)
    out.setdefault("seed", 42)
    out.setdefault("deterministic", True)
    out.setdefault("log_level", "INFO")
    return out



def create_batch_dict(
    data: Dict[str, np.ndarray],
    params: Dict[str, Any],
    idxs: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    if idxs is None:
        idxs = np.arange(data["x"].shape[0])
    if device is None:
        device = torch.device("cpu")

    batch: Dict[str, torch.Tensor] = {}
    batch["x"] = torch.as_tensor(data["x"][idxs], dtype=torch.float32, device=device)
    batch["dx"] = torch.as_tensor(data["dx"][idxs], dtype=torch.float32, device=device)

    if params["model_order"] == 2:
        batch["ddx"] = torch.as_tensor(data["ddx"][idxs], dtype=torch.float32, device=device)

    if params.get("sequential_thresholding", False):
        batch["coefficient_mask"] = torch.as_tensor(params["coefficient_mask"], dtype=torch.float32, device=device)

    batch["learning_rate"] = torch.as_tensor(params["learning_rate"], dtype=torch.float32, device=device)
    return batch



def _summarize_losses(losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {name: float(value.item()) for name, value in losses.items()}



def log_progress(
    model: SINDyAutoencoderModel,
    epoch: int,
    params: Dict[str, Any],
    train_batch: Dict[str, torch.Tensor],
    validation_batch: Dict[str, torch.Tensor],
    x_norm: float,
    sindy_predict_norm: float,
):
    with torch.no_grad():
        train_net = model(
            train_batch["x"],
            train_batch["dx"],
            train_batch.get("ddx", None),
            coefficient_mask=train_batch.get("coefficient_mask", None),
        )
        train_loss, train_losses, _ = define_loss_torch(train_net, params)

        val_net = model(
            validation_batch["x"],
            validation_batch["dx"],
            validation_batch.get("ddx", None),
            coefficient_mask=validation_batch.get("coefficient_mask", None),
        )
        val_loss, val_losses, _ = define_loss_torch(val_net, params)

    train_summary = _summarize_losses(train_losses)
    val_summary = _summarize_losses(val_losses)
    eps = 1e-12
    decoder_ratio = val_summary["decoder"] / max(abs(x_norm), eps)
    dynamics_ratio = val_summary["sindy_x"] / max(abs(sindy_predict_norm), eps)
    active_terms = int(np.sum(params["coefficient_mask"]))

    logger.info(
        "epoch=%d train_total=%.6e val_total=%.6e dec=%.6e lat=%.6e phys=%.6e reg=%.6e dec_ratio=%.6e dyn_ratio=%.6e active=%d",
        epoch,
        float(train_loss.item()),
        float(val_loss.item()),
        val_summary["decoder"],
        val_summary["sindy_z"],
        val_summary["sindy_x"],
        val_summary["sindy_regularization"],
        decoder_ratio,
        dynamics_ratio,
        active_terms,
    )
    logger.debug("train_losses=%s val_losses=%s", train_summary, val_summary)

    return np.array(
        (
            float(val_loss.item()),
            val_summary["decoder"],
            val_summary["sindy_z"],
            val_summary["sindy_x"],
            val_summary["sindy_regularization"],
        ),
        dtype=np.float32,
    )



def _save_training_artifacts(run_dir: Path, params: Dict[str, Any], model: SINDyAutoencoderModel, results_dict: Dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), run_dir / "model.pt")
    with (run_dir / "params.pkl").open("wb") as f:
        pickle.dump(params, f)
    with (run_dir / "metrics.pkl").open("wb") as f:
        pickle.dump(results_dict, f)

    validation_losses = np.asarray(results_dict.get("validation_losses", []), dtype=np.float32)
    if validation_losses.size > 0:
        np.save(run_dir / "validation_losses.npy", validation_losses)
        with (run_dir / "validation_losses.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["loss_total", "loss_decoder", "loss_sindy_z", "loss_sindy_x", "loss_sindy_regularization"])
            writer.writerows(validation_losses.tolist())

    np.save(run_dir / "sindy_model_terms.npy", np.asarray(results_dict.get("sindy_model_terms", []), dtype=np.int64))



def train_network_torch(
    training_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    params: ParamsLike,
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    run_dir = Path(run_dir) if run_dir is not None else None
    params_dict = coerce_params(params, run_dir=run_dir)
    device = resolve_device(params_dict.get("device", "cpu"))
    params_dict["device"] = device
    set_random_seed(int(params_dict["seed"]), bool(params_dict.get("deterministic", True)))

    logger.info(
        "starting training run_name=%s device=%s seed=%d deterministic=%s input_dim=%d latent_dim=%d model_order=%d poly_order=%d widths=%s batch_size=%d max_epochs=%d refinement_epochs=%d",
        params_dict.get("save_name", "model"),
        device,
        int(params_dict["seed"]),
        bool(params_dict.get("deterministic", True)),
        int(params_dict["input_dim"]),
        int(params_dict["latent_dim"]),
        int(params_dict["model_order"]),
        int(params_dict["poly_order"]),
        params_dict.get("widths", []),
        int(params_dict["batch_size"]),
        int(params_dict["max_epochs"]),
        int(params_dict["refinement_epochs"]),
    )
    logger.debug("full_params=%s", params_dict)

    learning_rate = float(params_dict["learning_rate"])
    model = SINDyAutoencoderModel(params_dict).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_refinement = torch.optim.Adam(model.parameters(), lr=learning_rate)

    validation_batch = create_batch_dict(val_data, params_dict, idxs=None, device=device)

    x_norm = float(np.mean(val_data["x"] ** 2))
    if params_dict["model_order"] == 1:
        sindy_predict_norm_x = float(np.mean(val_data["dx"] ** 2))
    else:
        sindy_predict_norm_x = float(np.mean(val_data["ddx"] ** 2))

    validation_losses = []
    sindy_model_terms = [int(np.sum(params_dict["coefficient_mask"]))]

    max_epochs = int(params_dict["max_epochs"])
    epoch_size = int(params_dict.get("epoch_size") or training_data["x"].shape[0])
    batch_size = int(params_dict["batch_size"])
    num_batches = epoch_size // batch_size
    logger.info("training_samples=%d validation_samples=%d num_batches=%d initial_active=%d", training_data["x"].shape[0], val_data["x"].shape[0], num_batches, sindy_model_terms[0])

    train_batch = None
    for epoch in range(max_epochs):
        model.train()
        for batch_idx in range(num_batches):
            idxs = np.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size)
            train_batch = create_batch_dict(training_data, params_dict, idxs=idxs, device=device)
            optimizer.zero_grad(set_to_none=True)
            network = model(
                train_batch["x"],
                train_batch["dx"],
                train_batch.get("ddx", None),
                coefficient_mask=train_batch.get("coefficient_mask", None),
            )
            loss, _, _ = define_loss_torch(network, params_dict)
            loss.backward()
            optimizer.step()

        if params_dict.get("print_progress", False) and (epoch % params_dict["print_frequency"] == 0):
            validation_losses.append(log_progress(model, epoch, params_dict, train_batch, validation_batch, x_norm, sindy_predict_norm_x))

        if (
            params_dict.get("sequential_thresholding", False)
            and (epoch % params_dict["threshold_frequency"] == 0)
            and (epoch > 0)
        ):
            with torch.no_grad():
                new_mask = (model.sindy_coefficients.detach().abs() > params_dict["coefficient_threshold"]).to(dtype=torch.float32, device=device)
                params_dict["coefficient_mask"] = new_mask.cpu().numpy()
                validation_batch["coefficient_mask"] = new_mask
                active_terms = int(new_mask.sum().item())
                logger.info("threshold_update epoch=%d threshold=%.6g active=%d", epoch, float(params_dict["coefficient_threshold"]), active_terms)
                sindy_model_terms.append(active_terms)

    logger.info("starting refinement phase")
    for refinement_epoch in range(int(params_dict["refinement_epochs"])):
        model.train()
        for batch_idx in range(num_batches):
            idxs = np.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size)
            train_batch = create_batch_dict(training_data, params_dict, idxs=idxs, device=device)
            optimizer_refinement.zero_grad(set_to_none=True)
            network = model(
                train_batch["x"],
                train_batch["dx"],
                train_batch.get("ddx", None),
                coefficient_mask=train_batch.get("coefficient_mask", None),
            )
            _, _, loss_refinement = define_loss_torch(network, params_dict)
            loss_refinement.backward()
            optimizer_refinement.step()

        if params_dict.get("print_progress", False) and (refinement_epoch % params_dict["print_frequency"] == 0):
            validation_losses.append(log_progress(model, refinement_epoch, params_dict, train_batch, validation_batch, x_norm, sindy_predict_norm_x))

    model.eval()
    with torch.no_grad():
        final_network = model(
            validation_batch["x"],
            validation_batch["dx"],
            validation_batch.get("ddx", None),
            coefficient_mask=validation_batch.get("coefficient_mask", None),
        )
        final_losses = define_loss_torch(final_network, params_dict)[1]
        if params_dict["model_order"] == 1:
            sindy_predict_norm_z = float(np.mean(final_network["dz"].detach().cpu().numpy() ** 2))
        else:
            sindy_predict_norm_z = float(np.mean(final_network["ddz"].detach().cpu().numpy() ** 2))
        sindy_coefficients = model.sindy_coefficients.detach().cpu().numpy()

    results_dict = {
        "num_epochs": max_epochs - 1 if max_epochs > 0 else 0,
        "x_norm": x_norm,
        "sindy_predict_norm_x": sindy_predict_norm_x,
        "sindy_predict_norm_z": sindy_predict_norm_z,
        "sindy_coefficients": sindy_coefficients,
        "loss_decoder": final_losses["decoder"].item(),
        "loss_decoder_sindy": final_losses["sindy_x"].item(),
        "loss_sindy": final_losses["sindy_z"].item(),
        "loss_sindy_regularization": final_losses["sindy_regularization"].item(),
        "validation_losses": np.array(validation_losses, dtype=np.float32),
        "sindy_model_terms": np.array(sindy_model_terms, dtype=np.int64),
    }

    logger.info(
        "training complete final_decoder=%.6e final_latent=%.6e final_physical=%.6e final_reg=%.6e active=%d",
        results_dict["loss_decoder"],
        results_dict["loss_sindy"],
        results_dict["loss_decoder_sindy"],
        results_dict["loss_sindy_regularization"],
        int(sindy_model_terms[-1]),
    )

    if run_dir is not None:
        _save_training_artifacts(run_dir, params_dict, model, results_dict)
        logger.info("saved training artifacts to %s", run_dir)

    return results_dict
