from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_tensor_like(value, ref: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(value, device=ref.device, dtype=ref.dtype)


def _activation_fn(name: str) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    if name == "relu":
        return F.relu
    if name == "elu":
        return F.elu
    if name == "sigmoid":
        return torch.sigmoid
    if name == "linear":
        return None
    raise ValueError("invalid activation function")


def sindy_library_torch(
    z: torch.Tensor,
    latent_dim: int,
    poly_order: int,
    include_sine: bool = False,
) -> torch.Tensor:
    library = [torch.ones(z.shape[0], device=z.device, dtype=z.dtype)]

    for i in range(latent_dim):
        library.append(z[:, i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                library.append(z[:, i] * z[:, j])

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    library.append(z[:, i] * z[:, j] * z[:, k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    for p in range(k, latent_dim):
                        library.append(z[:, i] * z[:, j] * z[:, k] * z[:, p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    for p in range(k, latent_dim):
                        for q in range(p, latent_dim):
                            library.append(z[:, i] * z[:, j] * z[:, k] * z[:, p] * z[:, q])

    if include_sine:
        for i in range(latent_dim):
            library.append(torch.sin(z[:, i]))

    return torch.stack(library, dim=1)


def sindy_library_torch_order2(
    z: torch.Tensor,
    dz: torch.Tensor,
    latent_dim: int,
    poly_order: int,
    include_sine: bool = False,
) -> torch.Tensor:
    library = [torch.ones(z.shape[0], device=z.device, dtype=z.dtype)]
    z_combined = torch.cat([z, dz], dim=1)

    for i in range(2 * latent_dim):
        library.append(z_combined[:, i])

    if poly_order > 1:
        for i in range(2 * latent_dim):
            for j in range(i, 2 * latent_dim):
                library.append(z_combined[:, i] * z_combined[:, j])

    if poly_order > 2:
        for i in range(2 * latent_dim):
            for j in range(i, 2 * latent_dim):
                for k in range(j, 2 * latent_dim):
                    library.append(z_combined[:, i] * z_combined[:, j] * z_combined[:, k])

    if poly_order > 3:
        for i in range(2 * latent_dim):
            for j in range(i, 2 * latent_dim):
                for k in range(j, 2 * latent_dim):
                    for p in range(k, 2 * latent_dim):
                        library.append(z_combined[:, i] * z_combined[:, j] * z_combined[:, k] * z_combined[:, p])

    if poly_order > 4:
        for i in range(2 * latent_dim):
            for j in range(i, 2 * latent_dim):
                for k in range(j, 2 * latent_dim):
                    for p in range(k, 2 * latent_dim):
                        for q in range(p, 2 * latent_dim):
                            library.append(
                                z_combined[:, i]
                                * z_combined[:, j]
                                * z_combined[:, k]
                                * z_combined[:, p]
                                * z_combined[:, q]
                            )

    if include_sine:
        for i in range(2 * latent_dim):
            library.append(torch.sin(z_combined[:, i]))

    return torch.stack(library, dim=1)


def z_derivative_torch(
    input_tensor: torch.Tensor,
    dx: torch.Tensor,
    weights: Sequence[torch.Tensor],
    biases: Sequence[torch.Tensor],
    activation: str = "elu",
) -> torch.Tensor:
    if len(weights) != len(biases):
        raise ValueError("weights and biases must have the same length")

    z = input_tensor
    dz = dx

    if activation == "elu":
        for i in range(len(weights) - 1):
            z = torch.matmul(z, weights[i]) + biases[i]
            dz = torch.minimum(torch.exp(z), torch.ones_like(z)) * torch.matmul(dz, weights[i])
            z = F.elu(z)
        dz = torch.matmul(dz, weights[-1])

    elif activation == "relu":
        for i in range(len(weights) - 1):
            z = torch.matmul(z, weights[i]) + biases[i]
            dz = (z > 0).to(z.dtype) * torch.matmul(dz, weights[i])
            z = F.relu(z)
        dz = torch.matmul(dz, weights[-1])

    elif activation == "sigmoid":
        for i in range(len(weights) - 1):
            z = torch.matmul(z, weights[i]) + biases[i]
            z = torch.sigmoid(z)
            dz = (z * (1 - z)) * torch.matmul(dz, weights[i])
        dz = torch.matmul(dz, weights[-1])

    else:
        for i in range(len(weights) - 1):
            dz = torch.matmul(dz, weights[i])
        dz = torch.matmul(dz, weights[-1])

    return dz


def z_derivative_order2_torch(
    input_tensor: torch.Tensor,
    dx: torch.Tensor,
    ddx: torch.Tensor,
    weights: Sequence[torch.Tensor],
    biases: Sequence[torch.Tensor],
    activation: str = "elu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(weights) != len(biases):
        raise ValueError("weights and biases must have the same length")

    z = input_tensor
    dz = dx
    ddz = ddx

    if activation == "elu":
        for i in range(len(weights) - 1):
            z = torch.matmul(z, weights[i]) + biases[i]
            dz_prev = torch.matmul(dz, weights[i])
            elu_derivative = torch.minimum(torch.exp(z), torch.ones_like(z))
            elu_derivative2 = torch.exp(z) * (z < 0).to(z.dtype)
            dz = elu_derivative * dz_prev
            ddz = elu_derivative2 * torch.square(dz_prev) + elu_derivative * torch.matmul(ddz, weights[i])
            z = F.elu(z)
        dz = torch.matmul(dz, weights[-1])
        ddz = torch.matmul(ddz, weights[-1])

    elif activation == "relu":
        for i in range(len(weights) - 1):
            z = torch.matmul(z, weights[i]) + biases[i]
            relu_derivative = (z > 0).to(z.dtype)
            dz = relu_derivative * torch.matmul(dz, weights[i])
            ddz = relu_derivative * torch.matmul(ddz, weights[i])
            z = F.relu(z)
        dz = torch.matmul(dz, weights[-1])
        ddz = torch.matmul(ddz, weights[-1])

    elif activation == "sigmoid":
        for i in range(len(weights) - 1):
            z = torch.matmul(z, weights[i]) + biases[i]
            z = torch.sigmoid(z)
            dz_prev = torch.matmul(dz, weights[i])
            sigmoid_derivative = z * (1 - z)
            sigmoid_derivative2 = sigmoid_derivative * (1 - 2 * z)
            dz = sigmoid_derivative * dz_prev
            ddz = sigmoid_derivative2 * torch.square(dz_prev) + sigmoid_derivative * torch.matmul(ddz, weights[i])
        dz = torch.matmul(dz, weights[-1])
        ddz = torch.matmul(ddz, weights[-1])

    else:
        for i in range(len(weights) - 1):
            dz = torch.matmul(dz, weights[i])
            ddz = torch.matmul(ddz, weights[i])
        dz = torch.matmul(dz, weights[-1])
        ddz = torch.matmul(ddz, weights[-1])

    return dz, ddz


class SINDyAutoencoderModel(nn.Module):
    """Canonical PyTorch runtime model. This is the only model implementation that
    should be used after the refactor. It preserves the validated manual-derivative
    pipeline from the professor's original TensorFlow code.
    """

    def __init__(self, params: Dict[str, Any]):
        super().__init__()

        self.input_dim = int(params["input_dim"])
        self.latent_dim = int(params["latent_dim"])
        self.activation_name = params["activation"]
        self.activation = _activation_fn(self.activation_name)
        self.poly_order = int(params["poly_order"])
        self.include_sine = bool(params.get("include_sine", False))
        self.model_order = int(params["model_order"])
        self.widths = list(params.get("widths", [])) if self.activation_name != "linear" else []

        self.encoder_dims = [self.input_dim] + self.widths + [self.latent_dim]
        self.decoder_dims = [self.latent_dim] + self.widths[::-1] + [self.input_dim]

        self.encoder_weights = nn.ParameterList()
        self.encoder_biases = nn.ParameterList()
        self.decoder_weights = nn.ParameterList()
        self.decoder_biases = nn.ParameterList()

        encoder_weight_override = params.get("encoder_weights", None)
        encoder_bias_override = params.get("encoder_biases", None)
        decoder_weight_override = params.get("decoder_weights", None)
        decoder_bias_override = params.get("decoder_biases", None)

        for i, (in_dim, out_dim) in enumerate(zip(self.encoder_dims[:-1], self.encoder_dims[1:])):
            if encoder_weight_override is not None and encoder_bias_override is not None:
                w = nn.Parameter(_to_tensor_like(encoder_weight_override[i], torch.empty((), dtype=torch.float32)))
                b = nn.Parameter(_to_tensor_like(encoder_bias_override[i], torch.empty((), dtype=torch.float32)))
            else:
                w = nn.Parameter(torch.empty(in_dim, out_dim))
                b = nn.Parameter(torch.zeros(out_dim))
                nn.init.xavier_uniform_(w)
            self.encoder_weights.append(w)
            self.encoder_biases.append(b)

        for i, (in_dim, out_dim) in enumerate(zip(self.decoder_dims[:-1], self.decoder_dims[1:])):
            if decoder_weight_override is not None and decoder_bias_override is not None:
                w = nn.Parameter(_to_tensor_like(decoder_weight_override[i], torch.empty((), dtype=torch.float32)))
                b = nn.Parameter(_to_tensor_like(decoder_bias_override[i], torch.empty((), dtype=torch.float32)))
            else:
                w = nn.Parameter(torch.empty(in_dim, out_dim))
                b = nn.Parameter(torch.zeros(out_dim))
                nn.init.xavier_uniform_(w)
            self.decoder_weights.append(w)
            self.decoder_biases.append(b)

        library_dim = int(params["library_dim"])
        coeff_override = params.get("sindy_coefficients", None)
        if coeff_override is not None:
            self.sindy_coefficients = nn.Parameter(torch.as_tensor(coeff_override, dtype=torch.float32).clone())
        else:
            init_mode = params["coefficient_initialization"]
            coeff = torch.empty(library_dim, self.latent_dim)
            if init_mode == "xavier":
                nn.init.xavier_uniform_(coeff)
            elif init_mode == "specified":
                coeff = torch.as_tensor(params["init_coefficients"], dtype=torch.float32).clone()
            elif init_mode == "constant":
                coeff.fill_(1.0)
            elif init_mode == "normal":
                coeff.normal_()
            else:
                raise ValueError("invalid coefficient_initialization")
            self.sindy_coefficients = nn.Parameter(coeff)

    def _forward_layers(
        self,
        x: torch.Tensor,
        weights: Sequence[torch.Tensor],
        biases: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        z = x
        for i in range(len(weights) - 1):
            z = torch.matmul(z, weights[i]) + biases[i]
            if self.activation is not None:
                z = self.activation(z)
        z = torch.matmul(z, weights[-1]) + biases[-1]
        return z

    def forward(
        self,
        x: torch.Tensor,
        dx: torch.Tensor,
        ddx: Optional[torch.Tensor] = None,
        coefficient_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        network: Dict[str, Any] = {}

        z = self._forward_layers(x, list(self.encoder_weights), list(self.encoder_biases))
        x_decode = self._forward_layers(z, list(self.decoder_weights), list(self.decoder_biases))

        if coefficient_mask is None:
            coefficient_mask = torch.ones_like(self.sindy_coefficients)

        if self.model_order == 1:
            dz = z_derivative_torch(x, dx, list(self.encoder_weights), list(self.encoder_biases), activation=self.activation_name)
            Theta = sindy_library_torch(z, self.latent_dim, self.poly_order, include_sine=self.include_sine)
            sindy_predict = torch.matmul(Theta, coefficient_mask * self.sindy_coefficients)
            dx_decode = z_derivative_torch(z, sindy_predict, list(self.decoder_weights), list(self.decoder_biases), activation=self.activation_name)
            network["dz_predict"] = sindy_predict
            network["dx_decode"] = dx_decode
        else:
            if ddx is None:
                raise ValueError("ddx must be provided when model_order == 2")
            dz, ddz = z_derivative_order2_torch(x, dx, ddx, list(self.encoder_weights), list(self.encoder_biases), activation=self.activation_name)
            Theta = sindy_library_torch_order2(z, dz, self.latent_dim, self.poly_order, include_sine=self.include_sine)
            sindy_predict = torch.matmul(Theta, coefficient_mask * self.sindy_coefficients)
            dx_decode, ddx_decode = z_derivative_order2_torch(z, dz, sindy_predict, list(self.decoder_weights), list(self.decoder_biases), activation=self.activation_name)
            network["ddz"] = ddz
            network["ddz_predict"] = sindy_predict
            network["ddx"] = ddx
            network["ddx_decode"] = ddx_decode

        network["x"] = x
        network["dx"] = dx
        network["z"] = z
        network["dz"] = dz
        network["x_decode"] = x_decode
        network["dx_decode"] = dx_decode
        network["encoder_weights"] = list(self.encoder_weights)
        network["encoder_biases"] = list(self.encoder_biases)
        network["decoder_weights"] = list(self.decoder_weights)
        network["decoder_biases"] = list(self.decoder_biases)
        network["Theta"] = Theta
        network["sindy_coefficients"] = self.sindy_coefficients
        return network
