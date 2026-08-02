from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class NormalizationStats:
    mean: float
    std: float

    @classmethod
    def from_mapping(cls, data: Dict[str, float]) -> "NormalizationStats":
        return cls(mean=float(data["mean"]), std=float(data["std"]))


def load_array(path: str | Path) -> np.ndarray:
    return np.load(Path(path))


def normalize_series(series: np.ndarray, stats: Optional[NormalizationStats] = None) -> Tuple[np.ndarray, NormalizationStats]:
    series = np.asarray(series, dtype=np.float32)
    if stats is None:
        mean = float(np.mean(series))
        std = float(np.std(series))
        if std == 0.0:
            std = 1.0
        stats = NormalizationStats(mean=mean, std=std)
    normalized = (series - stats.mean) / stats.std
    return normalized.astype(np.float32), stats


def denormalize_series(series: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return (np.asarray(series) * stats.std + stats.mean).astype(np.float32)


def save_normalization_stats(stats: NormalizationStats, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)


def load_normalization_stats(path: str | Path) -> NormalizationStats:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return NormalizationStats.from_mapping(data)


def compute_time_derivatives(series: np.ndarray, dt: float, model_order: int = 1) -> Dict[str, np.ndarray]:
    x = np.asarray(series, dtype=np.float32)
    dx = np.gradient(x, dt, axis=0).astype(np.float32)
    out: Dict[str, np.ndarray] = {"x": x, "dx": dx}
    if model_order == 2:
        ddx = np.gradient(dx, dt, axis=0).astype(np.float32)
        out["ddx"] = ddx
    return out


def slice_time_window(series: np.ndarray, window: Optional[Tuple[int, int]]) -> np.ndarray:
    if window is None:
        return np.asarray(series)
    start, end = window
    return np.asarray(series[start:end])


def prepare_kse_datasets(
    series: np.ndarray,
    dt: float,
    model_order: int,
    train_slice: Optional[Tuple[int, int]] = None,
    val_slice: Optional[Tuple[int, int]] = None,
    test_slice: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
    normalization_stats: Optional[NormalizationStats] = None,
) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]], Optional[NormalizationStats]]:
    stats = normalization_stats
    working = np.asarray(series, dtype=np.float32)

    if normalize:
        if stats is None:
            # Important: fit normalization only on the training window when available.
            # This avoids validation/test leakage.
            stats_source = slice_time_window(working, train_slice) if train_slice is not None else working
            _, stats = normalize_series(stats_source)
        working, stats = normalize_series(working, stats)

    train_data = compute_time_derivatives(slice_time_window(working, train_slice), dt, model_order) if train_slice else None
    val_data = compute_time_derivatives(slice_time_window(working, val_slice), dt, model_order) if val_slice else None
    test_data = compute_time_derivatives(slice_time_window(working, test_slice), dt, model_order) if test_slice else None
    return train_data, val_data, test_data, stats
