from __future__ import annotations

from typing import Any, Dict, Tuple

import torch


def _to_tensor_like(value, ref: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(value, device=ref.device, dtype=ref.dtype)


def define_loss_torch(network: Dict[str, Any], params: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    x = network["x"]
    x_decode = network["x_decode"]

    losses: Dict[str, torch.Tensor] = {}

    if params["model_order"] == 1:
        dz = network["dz"]
        dz_predict = network["dz_predict"]
        dx = network["dx"]
        dx_decode = network["dx_decode"]
    else:
        ddz = network["ddz"]
        ddz_predict = network["ddz_predict"]
        ddx = network["ddx"]
        ddx_decode = network["ddx_decode"]

    coefficient_mask = _to_tensor_like(params["coefficient_mask"], network["sindy_coefficients"])
    sindy_coefficients = coefficient_mask * network["sindy_coefficients"]

    losses["decoder"] = torch.mean((x - x_decode) ** 2)
    if params["model_order"] == 1:
        losses["sindy_z"] = torch.mean((dz - dz_predict) ** 2)
        losses["sindy_x"] = torch.mean((dx - dx_decode) ** 2)
    else:
        losses["sindy_z"] = torch.mean((ddz - ddz_predict) ** 2)
        losses["sindy_x"] = torch.mean((ddx - ddx_decode) ** 2)

    losses["sindy_regularization"] = torch.mean(torch.abs(sindy_coefficients))

    loss = (
        params["loss_weight_decoder"] * losses["decoder"]
        + params["loss_weight_sindy_z"] * losses["sindy_z"]
        + params["loss_weight_sindy_x"] * losses["sindy_x"]
        + params["loss_weight_sindy_regularization"] * losses["sindy_regularization"]
    )

    loss_refinement = (
        params["loss_weight_decoder"] * losses["decoder"]
        + params["loss_weight_sindy_z"] * losses["sindy_z"]
        + params["loss_weight_sindy_x"] * losses["sindy_x"]
    )

    return loss, losses, loss_refinement
