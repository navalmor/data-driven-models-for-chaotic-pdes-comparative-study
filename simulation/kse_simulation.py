"""
Kuramoto-Sivashinsky equation simulation used to generate the thesis dataset.

The solver uses a Fourier pseudo-spectral spatial discretization and a
semi-implicit time stepping scheme for the one-dimensional periodic KSE.
Runtime parameters are read from a JSON configuration file, and the saved
post-transient trajectory is used as the common physical dataset for the
AE-SINDy and forecasting experiments.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import json
import os
from pathlib import Path
import time
from typing import Any

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, ifft

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.logging_setup import add_logging_arguments, configure_from_args, get_logger
from common.plotting import add_time_axis_argument

logger = get_logger(__name__)


@dataclass
class KSESimulationConfig:
    output_dir: str = "simulation/simulation_data"

    nx: int = 64
    L: float = 22.0
    dt: float = 0.1
    n_steps: int = 30606
    transient: int = 500

    seed: int = 42
    out_every: int = 1

    no_dealias: bool = False
    verbose: bool = True

    save_plot: bool = True
    use_lyapunov: bool = True
    lambda_max: float = 0.043
    lyapunov_time_label: str = r"time $t$ [$1/\lambda_{\max}$]"
    physical_time_label: str = r"time $t$"

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> "KSESimulationConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)

        valid_keys = {field.name for field in fields(cls)}
        unknown_keys = sorted(set(raw) - valid_keys)
        if unknown_keys:
            raise ValueError(
                f"Unknown simulation config key(s) in {config_path}: {unknown_keys}"
            )

        return cls(**raw)


def simulate_kse(nx=128, L=60.0, dt=0.25, n_steps=50000, transient=100, seed=42, dealias=True, out_every=1, verbose=True):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, L, nx, endpoint=False)
    k = fftfreq(nx, d=L / nx) * 2.0 * np.pi
    k2 = k**2
    k4 = k**4
    L_hat = k2 - k4

    if dealias:
        kmax = np.max(np.abs(k))
        cutoff = (2.0/3.0) * kmax
        dealias_mask = np.abs(k) <= cutoff
    else:
        dealias_mask = np.ones_like(k, dtype=bool)

    u = np.cos(2.0 * np.pi * x / L) + 0.1 * rng.standard_normal(nx)
    u_hat = fft(u)

    def compute_N_hat(u_hat_local):
        u_phys = np.real(ifft(u_hat_local))
        u2 = u_phys * u_phys
        N_hat = -0.5j * k * fft(u2)
        if dealias:
            N_hat = N_hat * dealias_mask
        return N_hat

    u_series = []
    steps_saved = 0
    denom = 1.0 - 0.5 * dt * L_hat
    numer_lin = 1.0 + 0.5 * dt * L_hat
    N_hat_old = compute_N_hat(u_hat)
    denom_euler = 1.0 - dt * L_hat
    u_hat = (u_hat + dt * N_hat_old) / denom_euler
    N_hat = compute_N_hat(u_hat)

    if verbose:
        logger.info("Starting main loop (CN + AB2)...")

    for step in range(1, n_steps):
        AB2_term = dt * (1.5 * N_hat - 0.5 * N_hat_old)
        rhs = numer_lin * u_hat + AB2_term
        u_hat_new = rhs / denom
        N_hat_old = N_hat
        u_hat = u_hat_new
        N_hat = compute_N_hat(u_hat)

        if step >= transient and (step - transient) % out_every == 0:
            u_series.append(np.real(ifft(u_hat)))
            steps_saved += 1

        if verbose and (step % 5000 == 0 or step == n_steps-1):
            logger.info("PROGRESS | Step %d/%d", step, n_steps)

    u_series = np.array(u_series)
    return u_series, x


def save_spatiotemporal_plot(
    u_series,
    x,
    t,
    output_dir,
    use_lyapunov=False,
    lambda_max=0.043,
    lyapunov_time_label=r"time $t$ [$1/\lambda_{\max}$]",
    physical_time_label=r"time $t$",
    time_axis_mode=None,
    formats=("png",),
):
    logger = logging.getLogger(__name__)

    u_plot = np.asarray(u_series, dtype=float)
    x_plot = np.asarray(x, dtype=float)
    t_plot = np.asarray(t, dtype=float)

    # Display-only time axis. An explicit time_axis_mode overrides the legacy
    # use_lyapunov flag; the underlying field and samples are never changed.
    if time_axis_mode is not None:
        use_lyapunov = str(time_axis_mode).strip().lower() == "lyapunov"

    if use_lyapunov:
        t_plot = t_plot * float(lambda_max)
        xlabel = lyapunov_time_label
        stem = "kse_spatiotemporal_lyapunov"
    else:
        xlabel = physical_time_label
        stem = "kse_spatiotemporal"

    x_left = float(x_plot[0])
    x_right = float(x_plot[-1])
    if x_plot.size > 1:
        x_right = float(x_plot[-1] + (x_plot[1] - x_plot[0]))

    amplitude = float(np.nanmax(np.abs(u_plot)))
    if not np.isfinite(amplitude) or amplitude == 0.0:
        amplitude = 1.0

    from common import plot_style as _PS

    # Type sizes from the contract as well as the canvas: the 12/13/10 literals
    # this figure used to carry were chosen for a 10.5 in canvas and are not the
    # sizes a reader gets at the printed 5.9 in width.
    with plt.rc_context(_PS.rc_params("simulation_field")):
        fig, ax = plt.subplots(figsize=_PS.figure_size_in("simulation_field"),
                               constrained_layout=True)
        im = ax.imshow(
            u_plot.T,
            extent=[float(t_plot[0]), float(t_plot[-1]), x_left, x_right],
            aspect="auto",
            cmap="viridis",
            origin="lower",
            vmin=-amplitude,
            vmax=amplitude,
            interpolation="nearest",
            # Rasterised at figure dpi so the PDF carries the same image the PNG
            # shows, rather than the raw field array for the viewer to smooth.
            rasterized=True,
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"space $x$")
        ax.set_title(r"$u(x,t)$", loc="center", pad=8)

        # Colorbar carries no label: the mathematical panel identifier $u(x,t)$
        # already names the field (redundant-label removal). Tick values are kept.
        cbar = fig.colorbar(im, ax=ax, pad=0.02)

        # The extension used to be baked into the filename, so this generator
        # could only ever emit PNG and had no way to take part in the dual
        # PNG/PDF output the rest of the collection uses.
        os.makedirs(output_dir, exist_ok=True)
        written = []
        for fmt in formats:
            path = os.path.join(output_dir, f"{stem}.{str(fmt).lstrip('.')}")
            fig.savefig(path, dpi=_PS.style()["output"]["dpi"],
                        bbox_inches="tight", pad_inches=0.05)
            written.append(path)
        plt.close(fig)

    logger.info("saved spatiotemporal plot to %s", ", ".join(written))


def run_simulation(config: KSESimulationConfig, *, time_axis_mode=None):
    logger = logging.getLogger(__name__)
    start_time = time.time()

    logger.info(
        "starting KSE simulation nx=%d L=%.2f dt=%.3f n_steps=%d transient=%d",
        config.nx, config.L, config.dt, config.n_steps, config.transient
    )

    u_series, x = simulate_kse(
        nx=config.nx,
        L=config.L,
        dt=config.dt,
        n_steps=config.n_steps,
        transient=config.transient,
        seed=config.seed,
        dealias=not config.no_dealias,
        out_every=config.out_every,
        verbose=config.verbose,
    )

    logger.info(
        "Simulation output: shape=%s, dtype=%s", tuple(u_series.shape), u_series.dtype
    )

    n_time = u_series.shape[0]
    t = np.arange(n_time) * config.dt
    os.makedirs(config.output_dir, exist_ok=True)

    np.save(os.path.join(config.output_dir, "u_series.npy"), u_series)
    np.save(os.path.join(config.output_dir, "t.npy"), t)
    np.save(os.path.join(config.output_dir, "x.npy"), x)

    if config.save_plot:
        save_spatiotemporal_plot(
            u_series,
            x,
            t,
            config.output_dir,
            use_lyapunov=config.use_lyapunov,
            lambda_max=config.lambda_max,
            lyapunov_time_label=config.lyapunov_time_label,
            physical_time_label=config.physical_time_label,
            time_axis_mode=time_axis_mode,
        )

    logger.info(
        "saved data u_series=%s t=%s x=%s",
        u_series.shape, t.shape, x.shape
    )

    logger.info(
        "total runtime: %.2f seconds",
        time.time() - start_time
    )


def parse_args():
    parser = argparse.ArgumentParser(description="KSE Simulation")
    parser.add_argument("--config", required=True, help="Path to KSE simulation config JSON")
    add_time_axis_argument(parser)
    add_logging_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_from_args(args, component="KSE")
    config = KSESimulationConfig.from_json(args.config)
    run_simulation(config, time_axis_mode=args.time_axis_mode)


if __name__ == "__main__":
    main()
