from __future__ import annotations

import pickle
from collections import Counter
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.integrate import solve_ivp

from .configs import AESINDyConfig
from .logging_utils import get_logger
from .model import SINDyAutoencoderModel
from .train import create_batch_dict, resolve_device, set_random_seed
from .visualization import (
    plot_latent_comparison,
    plot_reconstruction,
    plot_rollout_encoder_comparison,
    plot_rollout_sindy_comparison,
    plot_xi_heatmap,
)

logger = get_logger("analysis")


def term_names_for_library(latent_dim: int, poly_order: int, include_sine: bool = False) -> List[str]:
    names = ["1"]
    for i in range(latent_dim):
        names.append(f"z{i+1}")

    def combo_name(combo):
        counts = Counter(combo)
        parts = []
        for idx in sorted(counts):
            power = counts[idx]
            base = f"z{idx+1}"
            if power == 1:
                parts.append(base)
            elif power == 2:
                parts.append(f"{base}²")
            elif power == 3:
                parts.append(f"{base}³")
            else:
                parts.append(f"{base}^{power}")
        return "·".join(parts)

    for degree in range(2, poly_order + 1):
        for combo in combinations_with_replacement(range(latent_dim), degree):
            names.append(combo_name(combo))

    if include_sine:
        for i in range(latent_dim):
            names.append(f"sin(z{i+1})")

    return names



def sindy_library_numpy(z: np.ndarray, latent_dim: int, poly_order: int, include_sine: bool = False) -> np.ndarray:
    z = np.asarray(z)
    library = [np.ones(z.shape[0], dtype=z.dtype)]
    for i in range(latent_dim):
        library.append(z[:, i])
    for degree in range(2, poly_order + 1):
        for combo in combinations_with_replacement(range(latent_dim), degree):
            term = np.ones(z.shape[0], dtype=z.dtype)
            for idx in combo:
                term *= z[:, idx]
            library.append(term)
    if include_sine:
        for i in range(latent_dim):
            library.append(np.sin(z[:, i]))
    return np.stack(library, axis=1)



def decode_numpy(z_array: np.ndarray, weights: Sequence[np.ndarray], biases: Sequence[np.ndarray], activation: str = "elu") -> np.ndarray:
    h = np.array(z_array, copy=True)
    for i, (w, b) in enumerate(zip(weights, biases)):
        h = h @ w + b
        if i < len(weights) - 1:
            if activation == "elu":
                h = np.where(h > 0, h, np.exp(np.clip(h, -50, 0)) - 1)
            elif activation == "sigmoid":
                h = 1 / (1 + np.exp(-np.clip(h, -50, 50)))
            elif activation == "relu":
                h = np.maximum(h, 0)
    return h



def sindy_ode(t, z_state, xi, poly_order, latent_dim, include_sine=False):
    theta = sindy_library_numpy(z_state[None, :], latent_dim, poly_order, include_sine=include_sine)[0]
    return theta @ xi



def load_config_from_run(run_dir: str | Path) -> AESINDyConfig:
    run_dir = Path(run_dir)
    pkl_path = run_dir / "params.pkl"
    json_path = run_dir / "config.json"
    if pkl_path.exists():
        return AESINDyConfig.from_pickle(pkl_path)
    if json_path.exists():
        return AESINDyConfig.from_json(json_path)
    raise FileNotFoundError(f"No params.pkl or config.json found in {run_dir}")



def find_candidate_checkpoint_paths(run_dir: Path, run_name: str) -> List[Path]:
    stem = run_dir / run_name
    candidates = [run_dir / "model.pt", run_dir / "model.pth", stem, stem.with_suffix(".pt"), stem.with_suffix(".pth")]
    for suffix in ["_refined", "_final", "_trained"]:
        candidates.append(Path(str(stem) + suffix))
        candidates.append(Path(str(stem) + suffix + ".pt"))
        candidates.append(Path(str(stem) + suffix + ".pth"))

    out: List[Path] = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out



def load_model_checkpoint(
    model: SINDyAutoencoderModel,
    run_dir: str | Path,
    run_name: str,
    strict: bool = True,
) -> Path:
    # H-01 safeguard: default to strict=True so a mismatched / partially-loaded
    # checkpoint fails loudly instead of silently running analysis on random
    # (xavier-initialised) weights. Pass strict=False only for known-legacy loads.
    run_dir = Path(run_dir)
    candidates = find_candidate_checkpoint_paths(run_dir, run_name)
    for path in candidates:
        if not path.exists():
            continue
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.nn.Module):
            model.load_state_dict(obj.state_dict(), strict=strict)
        elif isinstance(obj, dict):
            if "model_state_dict" in obj:
                model.load_state_dict(obj["model_state_dict"], strict=strict)
            elif "state_dict" in obj:
                model.load_state_dict(obj["state_dict"], strict=strict)
            else:
                model.load_state_dict(obj, strict=strict)
        else:
            raise TypeError(f"Unsupported PyTorch checkpoint type: {type(obj)}")
        model.eval()
        logger.info("loaded checkpoint from %s", path)
        return path

    tried = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No PyTorch checkpoint found. Tried:\n  {tried}")



def run_inference(model: SINDyAutoencoderModel, test_data: Dict[str, np.ndarray], params: Dict[str, Any], device: Optional[torch.device] = None) -> Dict[str, Any]:
    batch = create_batch_dict(test_data, params, idxs=None, device=device or torch.device("cpu"))
    device = device or next(model.parameters()).device
    x = batch["x"].to(device)
    dx = batch["dx"].to(device)
    ddx = batch.get("ddx", None)
    if ddx is not None:
        ddx = ddx.to(device)
    coefficient_mask = batch.get("coefficient_mask", None)
    if coefficient_mask is not None:
        coefficient_mask = coefficient_mask.to(device)
    with torch.no_grad():
        results = model(x, dx, ddx, coefficient_mask=coefficient_mask)
    # Keep analysis summaries CPU-loadable and lightweight.
    # The model returns lists of trainable weight/bias tensors for internal use;
    # they are not needed for scoring, plotting, or equation extraction.
    skip_keys = {
        "encoder_weights",
        "encoder_biases",
        "decoder_weights",
        "decoder_biases",
    }

    out = {}
    for key, value in results.items():
        if key in skip_keys:
            continue
        out[key] = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
    return out



def extract_equations(params: Dict[str, Any], test_set_results: Dict[str, Any]) -> Tuple[np.ndarray, List[str], List[str]]:
    Xi = params["coefficient_mask"] * test_set_results["sindy_coefficients"]
    term_names = term_names_for_library(params["latent_dim"], params["poly_order"], params.get("include_sine", False))
    equations = []
    for dim in range(params["latent_dim"]):
        active = np.where(params["coefficient_mask"][:, dim])[0]
        if len(active) == 0:
            equations.append(f"ż{dim+1} = 0")
            continue
        terms = [f"{Xi[idx, dim]:+.4f}·{term_names[idx]}" for idx in active]
        equations.append(f"ż{dim+1} = {' '.join(terms)}")
    return Xi, term_names, equations



def compute_relative_error_over_time(x_true: np.ndarray, x_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x_true = np.asarray(x_true, dtype=np.float64)
    x_pred = np.asarray(x_pred, dtype=np.float64)
    if x_true.shape != x_pred.shape:
        raise ValueError(f"Shape mismatch for error computation: {x_true.shape} vs {x_pred.shape}")
    numerator = np.linalg.norm(x_true - x_pred, axis=1)
    denominator = np.linalg.norm(x_true, axis=1) + eps
    return (numerator / denominator).astype(np.float64)



def compute_valid_horizon(t: np.ndarray, error_over_time: np.ndarray, threshold: float) -> Dict[str, Any]:
    t = np.asarray(t, dtype=np.float64)
    error_over_time = np.asarray(error_over_time, dtype=np.float64)
    if t.ndim != 1 or error_over_time.ndim != 1 or len(t) != len(error_over_time):
        raise ValueError("t and error_over_time must be one-dimensional arrays of the same length")
    if len(t) == 0:
        return {
            "valid_index": None,
            "valid_time": None,
            "valid_fraction": 0.0,
            "threshold_crossed": False,
            "first_invalid_index": None,
            "first_invalid_time": None,
        }

    # Invalid means either accuracy crossed threshold or error became NaN/Inf.
    invalid = (~np.isfinite(error_over_time)) | (error_over_time > threshold)
    invalid_indices = np.where(invalid)[0]
    threshold_crossed = len(invalid_indices) > 0

    if threshold_crossed:
        first_invalid_index = int(invalid_indices[0])
        first_invalid_time = float(t[first_invalid_index])

        # Important: the first invalid point itself is NOT valid.
        # If error crosses at step 1267, last valid step is 1266.
        valid_index = max(first_invalid_index - 1, 0)
    else:
        first_invalid_index = None
        first_invalid_time = None
        valid_index = len(t) - 1

    valid_time = float(t[valid_index])
    total_time = float(t[-1]) if len(t) > 1 else 0.0
    valid_fraction = valid_time / total_time if total_time > 0 else 1.0

    return {
        "valid_index": valid_index,
        "valid_time": valid_time,
        "valid_fraction": float(valid_fraction),
        "threshold_crossed": bool(threshold_crossed),
        "first_invalid_index": first_invalid_index,
        "first_invalid_time": first_invalid_time,
    }



def summarize_run_quality(
    decoder_error: float,
    sindy_valid_fraction: float,
    rollout_success: bool,
    rollout_used_fallback: bool,
    sparsity_percent: float,
) -> str:
    if not rollout_success:
        return "weak"
    if sindy_valid_fraction >= 0.8 and decoder_error <= 1.0 and sparsity_percent >= 10.0 and not rollout_used_fallback:
        return "good"
    if sindy_valid_fraction >= 0.4 and decoder_error <= 1.5:
        return "partial"
    return "weak"



def _usable_solution_prefix(
    sol: Any,
    explosion_limit: float = 1e6,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Return the finite/bounded prefix from a solve_ivp result.

    solve_ivp can return useful partial output even when sol.success is False.
    For horizon analysis, that partial trajectory is more informative than
    discarding the whole rollout and jumping to a very short fallback length.
    """
    if not hasattr(sol, "y") or sol.y is None or sol.y.size == 0:
        return None, None, "empty_solution"

    z_candidate = np.asarray(sol.y, dtype=np.float64).T
    t_candidate = np.asarray(sol.t, dtype=np.float64)

    n = min(len(t_candidate), z_candidate.shape[0])
    if n < 2:
        return None, None, "too_few_points"

    z_candidate = z_candidate[:n]
    t_candidate = t_candidate[:n]

    valid_rows = (
        np.all(np.isfinite(z_candidate), axis=1)
        & np.all(np.abs(z_candidate) <= explosion_limit, axis=1)
    )
    bad_indices = np.where(~valid_rows)[0]

    if len(bad_indices) > 0:
        first_bad = int(bad_indices[0])
        if first_bad < 2:
            return None, None, "invalid_immediately"
        return z_candidate[:first_bad], t_candidate[:first_bad], "truncated_invalid_values"

    if bool(getattr(sol, "success", False)):
        return z_candidate, t_candidate, "complete_solver_success"

    return z_candidate, t_candidate, "partial_solver_failure"


def simulate_latent_sindy(
    params: Dict[str, Any],
    Xi: np.ndarray,
    z0: np.ndarray,
    t_full: np.ndarray,
    decoder_weights: Sequence[np.ndarray],
    decoder_biases: Sequence[np.ndarray],
    fallback_steps: Sequence[int] = (),
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    requested_steps = int(len(t_full))
    rollout_meta = {
        "requested_steps": requested_steps,
        "used_steps": 0,
        "used_fallback": False,
        "rollout_success": False,
        "accepted_steps": None,
        "solver_success": False,
        "rollout_status": None,
    }

    if params["model_order"] != 1:
        logger.warning("latent SINDy simulation is implemented only for first-order systems")
        return None, None, None, rollout_meta

    if not fallback_steps:
        fallback_steps = (len(t_full), len(t_full) // 2, len(t_full) // 4, 110)

    for max_steps in fallback_steps:
        if max_steps < 2:
            continue

        t_try = t_full[:max_steps]

        try:
            sol = solve_ivp(
                sindy_ode,
                (t_try[0], t_try[-1]),
                z0,
                t_eval=t_try,
                args=(Xi, params["poly_order"], params["latent_dim"], params.get("include_sine", False)),
                method="RK45",
                rtol=1e-8,
                atol=1e-10,
                max_step=float(t_full[1] - t_full[0]) / 2 if len(t_full) > 1 else 0.05,
            )
        except Exception as exc:
            logger.debug("latent rollout attempt raised for steps=%d exception=%s", max_steps, exc)
            continue

        z_sim, t_sim, status = _usable_solution_prefix(sol)

        if z_sim is not None and t_sim is not None:
            x_sindy = decode_numpy(
                z_sim.astype(np.float32),
                decoder_weights,
                decoder_biases,
                params["activation"],
            )

            used_steps = int(len(t_sim))
            solver_success = bool(getattr(sol, "success", False))

            rollout_meta.update(
                {
                    "used_steps": used_steps,
                    "used_fallback": bool(max_steps != requested_steps or used_steps != requested_steps or not solver_success),
                    "rollout_success": True,
                    "accepted_steps": used_steps,
                    "solver_success": solver_success,
                    "rollout_status": status,
                }
            )

            logger.info(
                "usable latent SINDy rollout for %d/%d requested steps solver_success=%s status=%s",
                used_steps,
                max_steps,
                solver_success,
                status,
            )

            return z_sim, t_sim, x_sindy, rollout_meta

        logger.debug(
            "latent rollout attempt failed for steps=%d success=%s status=%s",
            max_steps,
            getattr(sol, "success", None),
            status,
        )

    logger.warning("latent SINDy simulation failed for all fallback horizons")
    return None, None, None, rollout_meta



def save_analysis_summary(
    run_dir: str | Path,
    summary: Dict[str, Any],
    split_label: Optional[str] = None,
    write_generic: bool = True,
) -> Path:
    # H-01 safeguard: support split-specific summary files
    # (analysis_summary_val.pkl / analysis_summary_test.pkl) so a validation-split
    # analysis never overwrites the generic/test analysis_summary.pkl.
    #   * split_label given  -> also write analysis_summary_<split_label>.pkl
    #   * write_generic=True -> write the legacy analysis_summary.pkl (default)
    # Default call (no args) preserves the original behaviour exactly.
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    primary: Optional[Path] = None
    if split_label:
        split_path = run_dir / f"analysis_summary_{split_label}.pkl"
        with split_path.open("wb") as f:
            pickle.dump(summary, f)
        primary = split_path

    if write_generic:
        out_path = run_dir / "analysis_summary.pkl"
        with out_path.open("wb") as f:
            pickle.dump(summary, f)
        if primary is None:
            primary = out_path

    return primary if primary is not None else (run_dir / "analysis_summary.pkl")



def run_analysis(
    run_dir: str | Path,
    test_data: Dict[str, np.ndarray],
    dt: float,
    device: str | torch.device = "auto",
    generate_plots: bool = False,
    rollout_max_steps: Optional[int] = None,
    eval_split: Optional[str] = None,
) -> Dict[str, Any]:
    # H-01 safeguard: `eval_split` ("val" | "test" | None) records which data split
    # `test_data` came from and controls summary-file naming:
    #   * "val"  -> writes analysis_summary_val.pkl only (never overwrites the
    #               generic/test analysis_summary.pkl); used for model SELECTION.
    #   * "test" -> writes analysis_summary.pkl (legacy) + analysis_summary_test.pkl;
    #               used for one-shot held-out REPORTING.
    #   * None   -> legacy behaviour (writes analysis_summary.pkl only).
    # The `test_data` argument may hold any split; the name is historical.
    run_dir = Path(run_dir)
    config = load_config_from_run(run_dir)
    params = config.to_params_dict(run_dir=run_dir)
    params["coefficient_mask"] = np.asarray(params["coefficient_mask"], dtype=np.float32)

    resolved_device = resolve_device(device)
    set_random_seed(config.seed, config.deterministic)

    logger.info(
        "starting analysis run_name=%s device=%s seed=%d deterministic=%s input_dim=%d latent_dim=%d model_order=%d poly_order=%d",
        config.run_name,
        resolved_device,
        config.seed,
        config.deterministic,
        config.input_dim,
        config.latent_dim,
        config.model_order,
        config.poly_order,
    )

    model = SINDyAutoencoderModel(params).to(resolved_device)
    checkpoint_path = load_model_checkpoint(model, run_dir, config.run_name)

    results = run_inference(model, test_data, params, device=resolved_device)
    Xi, term_names, equations = extract_equations(params, results)
    active_terms = int(np.sum(Xi != 0))
    sparsity_percent = 100.0 * (1.0 - active_terms / Xi.size)
    decoder_weights = [w.detach().cpu().numpy() for w in model.decoder_weights]
    decoder_biases = [b.detach().cpu().numpy() for b in model.decoder_biases]

    z_verify = results["z"][:5]
    x_numpy = decode_numpy(z_verify.astype(np.float32), decoder_weights, decoder_biases, params["activation"])
    x_model = results["x_decode"][:5]
    decoder_match_maxdiff = float(np.max(np.abs(x_numpy - x_model)))

    t_full = np.arange(results["z"].shape[0], dtype=np.float64) * dt
    fallback_steps = None
    if rollout_max_steps is not None:
        capped_steps = max(2, min(int(rollout_max_steps), len(t_full)))
        fallback_steps = (capped_steps, capped_steps // 2, capped_steps // 4, 110)

    z_sim, t_sim, x_sindy, rollout_meta = simulate_latent_sindy(
        params,
        Xi,
        results["z"][0],
        t_full,
        decoder_weights,
        decoder_biases,
        fallback_steps=() if fallback_steps is None else fallback_steps,
    )

    horizon_threshold = float(params.get("horizon_error_threshold", 0.5))
    decoder_error = float(np.mean((test_data["x"] - results["x_decode"]) ** 2) / np.mean(test_data["x"] ** 2))

    encoder_error_over_time = compute_relative_error_over_time(test_data["x"], results["x_decode"])
    encoder_horizon = compute_valid_horizon(t_full, encoder_error_over_time, horizon_threshold)

    encoder_rollout_error_over_time = None
    encoder_rollout_horizon = None
    sindy_error_over_time = None
    sindy_horizon = None
    if t_sim is not None and x_sindy is not None:
        n_sim = len(t_sim)
        encoder_rollout_error_over_time = compute_relative_error_over_time(test_data["x"][:n_sim], results["x_decode"][:n_sim])
        encoder_rollout_horizon = compute_valid_horizon(t_sim, encoder_rollout_error_over_time, horizon_threshold)
        sindy_error_over_time = compute_relative_error_over_time(test_data["x"][:n_sim], x_sindy)
        sindy_horizon = compute_valid_horizon(t_sim, sindy_error_over_time, horizon_threshold)

    requested_final_time = float(t_full[-1]) if len(t_full) > 0 else 0.0
    sindy_valid_fraction_requested = (
        float(sindy_horizon["valid_time"]) / requested_final_time
        if sindy_horizon is not None and requested_final_time > 0
        else 0.0
    )
    run_quality = summarize_run_quality(
        decoder_error=decoder_error,
        sindy_valid_fraction=sindy_valid_fraction_requested,
        rollout_success=bool(rollout_meta["rollout_success"]),
        rollout_used_fallback=bool(rollout_meta["used_fallback"]),
        sparsity_percent=float(sparsity_percent),
    )

    logger.info(
        "analysis summary checkpoint=%s decoder_error=%.6e decoder_match_maxdiff=%.6e active=%d sparsity=%.2f%% encoder_t_valid=%s sindy_t_valid=%s used_fallback=%s quality=%s",
        checkpoint_path,
        decoder_error,
        decoder_match_maxdiff,
        active_terms,
        sparsity_percent,
        None if encoder_horizon["valid_time"] is None else f"{encoder_horizon['valid_time']:.3f}",
        None if sindy_horizon is None or sindy_horizon["valid_time"] is None else f"{sindy_horizon['valid_time']:.3f}",
        rollout_meta["used_fallback"],
        run_quality,
    )
    logger.debug("equations=%s", equations)

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "equations": equations,
        "Xi": Xi,
        "term_names": term_names,
        "decoder_error": decoder_error,
        "decoder_match_maxdiff": decoder_match_maxdiff,
        "encoder_error_over_time": encoder_error_over_time,
        "encoder_mean_error": float(np.mean(encoder_error_over_time)),
        "encoder_max_error": float(np.max(encoder_error_over_time)),
        "encoder_valid_index": encoder_horizon["valid_index"],
        "encoder_valid_time": encoder_horizon["valid_time"],
        "encoder_valid_fraction": encoder_horizon["valid_fraction"],
        "encoder_first_invalid_index": encoder_horizon["first_invalid_index"],
        "encoder_first_invalid_time": encoder_horizon["first_invalid_time"],
        "encoder_threshold_crossed": encoder_horizon["threshold_crossed"],
        "encoder_rollout_error_over_time": encoder_rollout_error_over_time,
        "encoder_rollout_valid_index": None if encoder_rollout_horizon is None else encoder_rollout_horizon["valid_index"],
        "encoder_rollout_valid_time": None if encoder_rollout_horizon is None else encoder_rollout_horizon["valid_time"],
        "encoder_rollout_valid_fraction": None if encoder_rollout_horizon is None else encoder_rollout_horizon["valid_fraction"],
        "encoder_rollout_first_invalid_index": None if encoder_rollout_horizon is None else encoder_rollout_horizon["first_invalid_index"],
        "encoder_rollout_first_invalid_time": None if encoder_rollout_horizon is None else encoder_rollout_horizon["first_invalid_time"],
        "encoder_rollout_threshold_crossed": None if encoder_rollout_horizon is None else encoder_rollout_horizon["threshold_crossed"],
        "sindy_error_over_time": sindy_error_over_time,
        "sindy_mean_error": None if sindy_error_over_time is None else float(np.mean(sindy_error_over_time)),
        "sindy_max_error": None if sindy_error_over_time is None else float(np.max(sindy_error_over_time)),
        "sindy_valid_index": None if sindy_horizon is None else sindy_horizon["valid_index"],
        "sindy_valid_time": None if sindy_horizon is None else sindy_horizon["valid_time"],
        "sindy_valid_fraction": None if sindy_horizon is None else sindy_horizon["valid_fraction"],
        "sindy_first_invalid_index": None if sindy_horizon is None else sindy_horizon["first_invalid_index"],
        "sindy_first_invalid_time": None if sindy_horizon is None else sindy_horizon["first_invalid_time"],
        "sindy_threshold_crossed": None if sindy_horizon is None else sindy_horizon["threshold_crossed"],
        "horizon_error_threshold": horizon_threshold,
        "rollout_meta": rollout_meta,
        "active_terms": active_terms,
        "sparsity_percent": float(sparsity_percent),
        "run_quality": run_quality,
        "main_quality_metric": None if sindy_horizon is None else sindy_horizon["valid_time"],
        "z_sim": z_sim,
        "t_sim": t_sim,
        "x_sindy": x_sindy,
        "results": results,
    }

    # --- H-01 safeguard: clearer horizon-metric names + split-aware aliases ---
    # `sindy_valid_time` means "prediction-validity horizon", NOT "validation-set".
    # The old keys are kept for backward compatibility with historical artifacts
    # and readers; new keys disambiguate the metric and the data split.
    summary["sindy_horizon_time"] = summary["sindy_valid_time"]
    summary["encoder_horizon_time"] = summary["encoder_valid_time"]
    summary["horizon_metric_name"] = "sindy_rollout_horizon_time"
    summary["metric_name"] = "sindy_rollout_horizon_time"
    summary["eval_split"] = eval_split
    summary["test_used_for_selection"] = False
    if eval_split in ("val", "test"):
        summary[f"{eval_split}_sindy_horizon_time"] = summary["sindy_valid_time"]
        summary[f"{eval_split}_decoder_error"] = summary["decoder_error"]

    save_analysis_summary(
        run_dir,
        summary,
        split_label=eval_split,
        write_generic=(eval_split != "val"),
    )

    if generate_plots:
        plot_dir = run_dir / "plots"
        plot_reconstruction(
            test_data,
            results,
            dt,
            horizon_time=summary["encoder_valid_time"],
            save_dir=plot_dir,
            show=False,
        )
        plot_xi_heatmap(results, Xi, term_names, save_dir=plot_dir, show=False)
        if z_sim is not None and t_sim is not None:
            plot_latent_comparison(params, results, z_sim, t_sim, save_dir=plot_dir, show=False)
            plot_rollout_encoder_comparison(
                test_data,
                results,
                t_sim,
                horizon_time=summary["encoder_rollout_valid_time"],
                save_dir=plot_dir,
                show=False,
            )
        if x_sindy is not None and t_sim is not None:
            plot_rollout_sindy_comparison(
                test_data,
                x_sindy,
                t_sim,
                horizon_time=summary["sindy_valid_time"],
                save_dir=plot_dir,
                show=False,
            )
        logger.info("saved analysis plots to %s", plot_dir)

    return summary
