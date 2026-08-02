from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .sindy_utils import library_size


@dataclass
class AESINDyConfig:
    # Model
    input_dim: int
    latent_dim: int
    model_order: int = 1
    poly_order: int = 2
    include_sine: bool = False
    activation: str = "elu"
    widths: List[int] = field(default_factory=lambda: [128, 64])

    # Coefficients / sparsity
    coefficient_initialization: str = "constant"
    sequential_thresholding: bool = True
    coefficient_threshold: float = 0.1
    coefficient_mask: Optional[Any] = None
    init_coefficients: Optional[Any] = None

    # Loss weights
    loss_weight_decoder: float = 1.0
    loss_weight_sindy_z: float = 0.1
    loss_weight_sindy_x: float = 0.1
    loss_weight_sindy_regularization: float = 1e-4

    # Training
    learning_rate: float = 1e-4
    batch_size: int = 128
    epoch_size: Optional[int] = None
    max_epochs: int = 1000
    refinement_epochs: int = 1000
    print_progress: bool = True
    print_frequency: int = 10
    threshold_frequency: int = 100
    seed: int = 42
    deterministic: bool = True

    # Runtime / artifacts
    device: str = "auto"
    output_dir: str = "experiments"
    run_name: str = "model1"
    log_level: str = "INFO"

    # Data
    dt: float = 0.1
    data_file: Optional[str] = None
    train_slice: Optional[Tuple[int, int]] = None
    val_slice: Optional[Tuple[int, int]] = None
    test_slice: Optional[Tuple[int, int]] = None
    normalize: bool = True

    # Analysis
    horizon_error_threshold: float = 0.5
    analysis_generate_plots: bool = False
    analysis_rollout_max_steps: Optional[int] = None

    # Visualization
    visualization_output_dir: Optional[str] = None
    visualization_show: bool = False
    use_lyapunov: bool = False
    lambda_max: float = 1.0
    lyapunov_time_label: str = r"time $t$ [$1/\lambda_{\max}$]"
    physical_time_label: str = r"time $t$"

    def __post_init__(self) -> None:
        self.input_dim = int(self.input_dim)
        self.latent_dim = int(self.latent_dim)
        self.model_order = int(self.model_order)
        self.poly_order = int(self.poly_order)
        self.widths = [int(w) for w in self.widths]
        self.batch_size = int(self.batch_size)
        self.max_epochs = int(self.max_epochs)
        self.refinement_epochs = int(self.refinement_epochs)
        self.print_frequency = int(self.print_frequency)
        self.threshold_frequency = int(self.threshold_frequency)
        self.seed = int(self.seed)
        self.log_level = str(self.log_level).upper()
        self.horizon_error_threshold = float(self.horizon_error_threshold)
        if isinstance(self.use_lyapunov, str):
            self.use_lyapunov = self.use_lyapunov.strip().lower() in {"1", "true", "yes", "on"}
        else:
            self.use_lyapunov = bool(self.use_lyapunov)
        self.lambda_max = float(self.lambda_max)
        if self.lambda_max <= 0:
            raise ValueError("lambda_max must be positive")
        self.lyapunov_time_label = str(self.lyapunov_time_label)
        self.physical_time_label = str(self.physical_time_label)

    @property
    def library_dim(self) -> int:
        n = self.latent_dim if self.model_order == 1 else 2 * self.latent_dim
        return int(library_size(n, self.poly_order, self.include_sine, include_constant=True))

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name

    def make_coefficient_mask(self) -> np.ndarray:
        if self.coefficient_mask is None:
            return np.ones((self.library_dim, self.latent_dim), dtype=np.float32)
        return np.asarray(self.coefficient_mask, dtype=np.float32)

    def to_params_dict(self, run_dir: Optional[Path] = None) -> Dict[str, Any]:
        save_dir = Path(run_dir) if run_dir is not None else self.run_dir
        params = {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "model_order": self.model_order,
            "poly_order": self.poly_order,
            "include_sine": self.include_sine,
            "library_dim": self.library_dim,
            "activation": self.activation,
            "widths": list(self.widths),
            "coefficient_initialization": self.coefficient_initialization,
            "sequential_thresholding": self.sequential_thresholding,
            "coefficient_threshold": self.coefficient_threshold,
            "coefficient_mask": self.make_coefficient_mask(),
            "loss_weight_decoder": self.loss_weight_decoder,
            "loss_weight_sindy_z": self.loss_weight_sindy_z,
            "loss_weight_sindy_x": self.loss_weight_sindy_x,
            "loss_weight_sindy_regularization": self.loss_weight_sindy_regularization,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epoch_size": self.epoch_size,
            "max_epochs": self.max_epochs,
            "refinement_epochs": self.refinement_epochs,
            "print_progress": self.print_progress,
            "print_frequency": self.print_frequency,
            "threshold_frequency": self.threshold_frequency,
            "device": self.device,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "data_path": str(save_dir) + "/",
            "save_name": self.run_name,
            "dt": self.dt,
            "data_file": self.data_file,
            "train_slice": self.train_slice,
            "val_slice": self.val_slice,
            "test_slice": self.test_slice,
            "normalize": self.normalize,
            "horizon_error_threshold": self.horizon_error_threshold,
            "analysis_generate_plots": self.analysis_generate_plots,
            "analysis_rollout_max_steps": self.analysis_rollout_max_steps,
            "visualization_output_dir": self.visualization_output_dir,
            "visualization_show": self.visualization_show,
            "use_lyapunov": self.use_lyapunov,
            "lambda_max": self.lambda_max,
            "lyapunov_time_label": self.lyapunov_time_label,
            "physical_time_label": self.physical_time_label,
            "log_level": self.log_level,
        }
        if self.init_coefficients is not None:
            params["init_coefficients"] = np.asarray(self.init_coefficients, dtype=np.float32)
        return params

    def to_serializable_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["library_dim"] = self.library_dim
        if data["coefficient_mask"] is not None:
            data["coefficient_mask"] = np.asarray(data["coefficient_mask"]).tolist()
        if data["init_coefficients"] is not None:
            data["init_coefficients"] = np.asarray(data["init_coefficients"]).tolist()
        return data

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_serializable_dict(), f, indent=2)

    def save_pickle(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.to_params_dict(path.parent), f)

    @classmethod
    def from_json(cls, path: str | Path) -> "AESINDyConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_mapping(data)

    @classmethod
    def from_pickle(cls, path: str | Path) -> "AESINDyConfig":
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AESINDyConfig":
        data = dict(data)
        if "save_name" in data and "run_name" not in data:
            data["run_name"] = Path(str(data["save_name"])).name
        if "data_path" in data and "output_dir" not in data:
            data_path = Path(str(data["data_path"])).resolve()
            data["output_dir"] = str(data_path.parent)
        if "widths" in data and data["widths"] is not None:
            if isinstance(data["widths"], np.ndarray):
                data["widths"] = data["widths"].tolist()
            elif isinstance(data["widths"], tuple):
                data["widths"] = list(data["widths"])
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)
