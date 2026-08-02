from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def resolve_path(path: str | Path, repo_root: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (Path(repo_root).resolve() / p).resolve()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise TypeError(f"JSON file must contain an object: {path}")

    return obj


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def prepare_output_dir(path: str | Path, *, overwrite: bool) -> Path:
    path = Path(path)

    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory exists and overwrite=false: {path}")
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def discover_optimizer_result_files(search_dir: str | Path) -> list[tuple[str, Path]]:
    """
    Return (optimizer_name, optimizer_results.csv) pairs from one optimizer-search directory.

    Expected layout:
        search_dir/
          gp/optimizer_results.csv
          dummy/optimizer_results.csv
          forest/optimizer_results.csv
          gbrt/optimizer_results.csv
    """
    search_dir = Path(search_dir)

    if not search_dir.exists():
        raise FileNotFoundError(f"Optimizer search directory does not exist: {search_dir}")

    pairs: list[tuple[str, Path]] = []

    for child in sorted(search_dir.iterdir()):
        if not child.is_dir():
            continue

        results_path = child / "optimizer_results.csv"
        if results_path.exists():
            pairs.append((child.name, results_path))

    if not pairs:
        raise FileNotFoundError(
            f"No */optimizer_results.csv files found under optimizer search directory: {search_dir}"
        )

    return pairs


def load_optimizer_results(search_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for search_dir in search_dirs:
        for optimizer_name, results_path in discover_optimizer_result_files(search_dir):
            for raw in read_csv_rows(results_path):
                row: dict[str, Any] = dict(raw)
                row["optimizer"] = optimizer_name
                row["search_dir"] = str(search_dir)
                row["results_path"] = str(results_path)
                rows.append(row)

    if not rows:
        raise ValueError("Loaded zero optimizer rows.")

    return rows


def load_split_arrays(run_dir: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    split_dir = Path(run_dir) / split
    truth_path = split_dir / "truth_physical.npy"
    pred_path = split_dir / "pred_physical.npy"

    if not truth_path.exists() or not pred_path.exists():
        missing = [str(p) for p in [truth_path, pred_path] if not p.exists()]
        raise FileNotFoundError(f"Missing physical arrays for split={split}: {missing}")

    truth = np.load(truth_path)
    pred = np.load(pred_path)

    if truth.shape != pred.shape:
        raise ValueError(f"truth/pred shape mismatch for {split_dir}: {truth.shape} vs {pred.shape}")

    if truth.ndim != 2:
        raise ValueError(f"Expected 2D arrays shaped (time, state), got {truth.shape}")

    return truth, pred


def load_split_errors(run_dir: str | Path, split: str) -> list[dict[str, str]]:
    path = Path(run_dir) / split / "errors_physical.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing split errors CSV: {path}")

    return read_csv_rows(path)


def load_split_metrics(run_dir: str | Path, split: str) -> dict[str, Any]:
    path = Path(run_dir) / split / "metrics_physical.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing split metrics JSON: {path}")

    return read_json(path)


def load_run_summary(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "summary.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing run summary: {path}")

    return read_json(path)


def load_run_config(run_dir: str | Path) -> dict[str, Any]:
    """
    Return the resolved configuration a run was executed with.

    This is the record of the settings that produced the run's arrays and
    metrics, so it is the authoritative source for a value a leaner metrics
    schema omits. Read-only: nothing here writes or mutates the record.
    """
    path = Path(run_dir) / "config_resolved.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing resolved run config: {path}")

    return read_json(path)
