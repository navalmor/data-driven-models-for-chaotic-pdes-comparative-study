from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from .configs import AESINDyConfig
from .data_utils import load_array, load_normalization_stats, prepare_kse_datasets
from .logging_utils import get_logger

logger = get_logger("visualization")


def _cp():
    """Lazy accessor for the shared plotting helpers.

    ``common`` is imported inside the call (not at module import time) so this
    library package never assumes the repository root is on ``sys.path`` -- the
    entry points bootstrap it, mirroring ``ae_sindy.logging_utils``.
    """
    from common import plotting

    return plotting


def load_config_from_run(run_dir: str | Path) -> AESINDyConfig:
    run_dir = Path(run_dir)
    pkl_path = run_dir / "params.pkl"
    json_path = run_dir / "config.json"
    if pkl_path.exists():
        return AESINDyConfig.from_pickle(pkl_path)
    if json_path.exists():
        return AESINDyConfig.from_json(json_path)
    raise FileNotFoundError(f"Neither params.pkl nor config.json found in {run_dir}")



def _ensure_dir(path: Optional[str | Path]) -> Optional[Path]:
    if path is None:
        return None
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out



def load_run_artifacts(run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    artifacts: Dict[str, Any] = {"run_dir": run_dir}

    metrics_path = run_dir / "metrics.pkl"
    if metrics_path.exists():
        with metrics_path.open("rb") as f:
            artifacts["metrics"] = pickle.load(f)

    analysis_path = run_dir / "analysis_summary.pkl"
    if analysis_path.exists():
        with analysis_path.open("rb") as f:
            artifacts["analysis_summary"] = pickle.load(f)

    validation_losses_csv = run_dir / "validation_losses.csv"
    validation_losses_npy = run_dir / "validation_losses.npy"
    if validation_losses_npy.exists():
        artifacts["validation_losses"] = np.load(validation_losses_npy)
    elif validation_losses_csv.exists():
        artifacts["validation_losses"] = np.loadtxt(validation_losses_csv, delimiter=",", skiprows=1)

    model_terms_path = run_dir / "sindy_model_terms.npy"
    if model_terms_path.exists():
        artifacts["sindy_model_terms"] = np.load(model_terms_path)

    return artifacts



def load_test_data_for_run(run_dir: str | Path, data_file: Optional[str] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    cfg = load_config_from_run(run_dir)
    cfg_data_file = data_file or cfg.data_file
    if not cfg_data_file:
        raise ValueError("A data file is required either in the saved config or via data_file")
    stats = None
    if cfg.normalize:
        stats_path = Path(run_dir) / "normalization_stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Missing {stats_path}. Re-train this run with train-fitted normalization before visualization."
            )
        stats = load_normalization_stats(stats_path)

    series = load_array(cfg_data_file)
    _, _, test_data, stats = prepare_kse_datasets(
        series,
        dt=cfg.dt,
        model_order=cfg.model_order,
        train_slice=None,
        val_slice=None,
        test_slice=cfg.test_slice,
        normalize=cfg.normalize,
        normalization_stats=stats,
    )
    if test_data is None:
        raise ValueError("Saved config does not define test_slice; cannot rebuild visualization test data")
    return test_data, stats



def _ps():
    """Lazy accessor for the central style contract (see ``_cp``).

    ``common`` is imported inside the call, not at module import time, so this
    library package never assumes the repository root is on ``sys.path``.
    """
    from common import plot_style

    return plot_style


def _add_horizon_marker(ax, horizon_time: Optional[float], crossed: bool = True,
                        *, quantity: str = "reconstruction_validity_time",
                        use_lyapunov: bool = True) -> None:
    r"""Mark a validity time, naming the quantity it actually measures.

    Two scientifically different quantities are drawn by this module and both
    were previously labelled ``t_valid``:

      ``prediction_horizon``            the decoded latent SINDy rollout against
                                        the truth -- a forecast, and the same
                                        construct as the forecasting chapter's
                                        prediction horizon
      ``reconstruction_validity_time``  the autoencoder round trip, truth against
                                        encode-then-decode. No dynamics are
                                        involved, so it is NOT a forecast horizon
                                        and must not borrow that name.

    The two pipelines also differ in indexing and warm-up handling; those
    differences are documented, and deliberately not changed, in
    ``docs/horizon_quantities.md``.

    A value that never crossed the threshold inside the displayed window is a
    lower bound rather than an estimate, so it keeps the ``\geq`` relation.
    """
    if horizon_time is None:
        return
    from common import plot_style as _PS

    hor = _PS.marker("prediction_horizon")
    symbol = _PS.horizon_symbol(use_lyapunov)
    relation = "=" if crossed else r"\geq"
    ax.axvline(
        horizon_time,
        color=hor["color"], linestyle=hor["linestyle"],
        linewidth=hor["linewidth"], alpha=hor["alpha"],
        label=f"{_PS.term(quantity)}, ${symbol} {relation} {horizon_time:.3f}$",
    )


def _time_axis_from_config(cfg: AESINDyConfig, time_axis_mode: Optional[str] = None) -> Tuple[float, str]:
    # Display-only time axis. An explicit time_axis_mode (from the CLI) overrides
    # the config's legacy use_lyapunov flag; otherwise use_lyapunov selects the
    # mode. Precedence: CLI > config use_lyapunov > lyapunov default.
    if time_axis_mode is not None:
        mode = _cp().resolve_time_axis_mode(cli_value=time_axis_mode)
    else:
        mode = "lyapunov" if bool(getattr(cfg, "use_lyapunov", False)) else "physical"

    if mode == "lyapunov":
        return float(getattr(cfg, "lambda_max", 1.0)), str(getattr(cfg, "lyapunov_time_label", r"time $t$ [$1/\lambda_{\max}$]"))
    return 1.0, str(getattr(cfg, "physical_time_label", r"time $t$"))


def _scale_horizon_time(horizon_time: Optional[float], time_scale: float) -> Optional[float]:
    if horizon_time is None:
        return None
    return float(horizon_time) * float(time_scale)




def _plot_field_triptych_vertical(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    time_grid: np.ndarray,
    *,
    title_prefix: str,
    pred_title: str,
    horizon_time: Optional[float] = None,
    horizon_crossed: bool = True,
    horizon_quantity: str = "reconstruction_validity_time",
    save_path: Optional[Path] = None,
    show: bool = True,
    x_label: str = "Time",
    true_title: str = r"$u(x,t)$",
    error_title: Optional[str] = None,
    true_colorbar_label: str = r"$u(x,t)$",
    pred_colorbar_label: str = r"$\hat{u}(x,t)$",
    error_colorbar_label: Optional[str] = None,
    field_scale_mode: str = "shared",
    error_scale_mode: str = "residual",
) -> None:
    """Plot a three-panel field comparison."""
    spatial_grid = _cp().kse_spatial_grid(x_true.shape[1])

    # Panels (a)/(b) share one symmetric range derived from the displayed TRUE field
    # only, and panel (c) gets an independent symmetric error range. Both come from the
    # single repository-wide helper in common.plotting, so this triptych follows the
    # same convention as the forecasting rollout triptychs.
    #
    # The former ``field_scale_mode``/``error_scale_mode`` switches are retained as
    # accepted keywords for call-site compatibility, but they no longer select a
    # scaling rule: mixing conventions across figures is exactly what this
    # standardisation removes. Any value other than the compliant one is rejected
    # rather than silently honoured.
    if field_scale_mode not in ("shared", "true_symmetric"):
        raise ValueError(f"Unknown field_scale_mode: {field_scale_mode!r}")
    if error_scale_mode not in ("residual", "field"):
        raise ValueError(f"Unknown error_scale_mode: {error_scale_mode!r}")

    field_vmin, field_vmax = _cp().truth_prediction_limits(x_true)
    error_vmin, error_vmax = _cp().error_limits(x_true, x_pred)

    signed_error = x_true - x_pred

    if error_title is None:
        error_title = r"$u(x,t)-\hat{u}(x,t)$"
    if error_colorbar_label is None:
        error_colorbar_label = error_title

    with plt.rc_context(_ps().rc_params("ae_sindy_field_triptych")):
        fig, axes = plt.subplots(
            3,
            1,
            figsize=_ps().figure_size_in("ae_sindy_field_triptych"),
            sharex=True,
            constrained_layout=True,
        )

        # Titles carry the mathematical field identifier; the a/b/c panel
        # label is drawn inside each axes below.
        panel_titles = [true_title, pred_title, error_title]

        im0 = axes[0].pcolormesh(
            time_grid,
            spatial_grid,
            x_true.T,
            cmap="viridis",
            shading="auto",
            vmin=field_vmin,
            vmax=field_vmax,
            rasterized=True,
        )
        axes[0].set_title(panel_titles[0], loc="left", x=0.35, pad=8)
        axes[0].set_ylabel(r"space $x$")
        # Colorbar carries no label: the panel identifier already names the field.
        fig.colorbar(im0, ax=axes[0], pad=0.025, fraction=0.045)

        im1 = axes[1].pcolormesh(
            time_grid,
            spatial_grid,
            x_pred.T,
            cmap="viridis",
            shading="auto",
            vmin=field_vmin,
            vmax=field_vmax,
            rasterized=True,
        )
        axes[1].set_title(panel_titles[1], loc="left", x=0.35, pad=8)
        axes[1].set_ylabel(r"space $x$")
        fig.colorbar(im1, ax=axes[1], pad=0.025, fraction=0.045)

        im2 = axes[2].pcolormesh(
            time_grid,
            spatial_grid,
            signed_error.T,
            cmap="RdBu_r",
            shading="auto",
            vmin=error_vmin,
            vmax=error_vmax,
            rasterized=True,
        )
        axes[2].set_title(panel_titles[2], loc="left", x=0.35, pad=8)
        axes[2].set_xlabel(x_label)
        axes[2].set_ylabel(r"space $x$")
        fig.colorbar(im2, ax=axes[2], pad=0.025, fraction=0.045)

        for ax in axes:
            ax.set_ylim(spatial_grid[0], spatial_grid[-1])
            ax.grid(False)
            _add_horizon_marker(ax, horizon_time, horizon_crossed,
                                quantity=horizon_quantity)

        _cp().add_panel_labels(axes)

        if horizon_time is not None:
            axes[2].legend(loc="upper right", frameon=True)

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
            logger.info("saved plot to %s", save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)




def plot_training_history(
    validation_losses: np.ndarray,
    sindy_model_terms: Optional[np.ndarray] = None,
    *,
    save_dir: Optional[str | Path] = None,
    show: bool = True,
    filename: str = "training_history.png",
    title_prefix: str = "e",
) -> None:
    """Plot validation loss components and active SINDy terms."""
    if validation_losses is None or len(validation_losses) == 0:
        logger.warning("No validation losses available for plotting")
        return

    validation_losses = np.asarray(validation_losses)
    save_dir = _ensure_dir(save_dir)

    epochs = np.arange(len(validation_losses))
    loss_labels = [
        r"$\mathcal{L}_{\mathrm{total}}$",
        r"$\mathcal{L}_{\mathrm{dec}}$",
        r"$\mathcal{L}_{z}$",
        r"$\mathcal{L}_{x}$",
        r"$\mathcal{L}_{\mathrm{reg}}$",
    ]

    # Type sizes from the contract. The 13 pt title literal these panels used to
    # carry was sized for an 11.9 in canvas; at the printed 5.9 in width it made
    # "validation loss components" 65 px wider than its own panel, so the title
    # ran into the neighbouring panel's y-axis label.
    with plt.rc_context(_ps().rc_params("ae_sindy_training_history")):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=_ps().figure_size_in("ae_sindy_training_history"),
            constrained_layout=True,
        )

        for i, label in enumerate(loss_labels):
            if i < validation_losses.shape[1]:
                vals = np.asarray(validation_losses[:, i], dtype=float)
                vals = np.where(vals > 0, vals, np.nan)
                axes[0].plot(epochs, vals, linewidth=1.8, label=label)

        axes[0].set_title(
            "validation loss components",
            loc="left",
            x=0.10,
            pad=8,
        )
        axes[0].set_xlabel("validation checkpoint")
        axes[0].set_ylabel("validation loss")
        axes[0].set_yscale("log")
        axes[0].grid(True, which="both", alpha=0.25)
        axes[0].legend(loc="upper right", frameon=True, ncol=1)

        if sindy_model_terms is not None and len(sindy_model_terms) > 0:
            active_terms = np.asarray(sindy_model_terms, dtype=float)
            term_steps = np.arange(len(active_terms))
            axes[1].step(
                term_steps,
                active_terms,
                where="post",
                linewidth=2.0,
                label="active terms",
            )
            axes[1].set_title(
                "active SINDy terms",
                loc="left",
                x=0.10,
                pad=8,
            )
            axes[1].set_xlabel("threshold step")
            axes[1].set_ylabel("number of active terms")
            axes[1].grid(True, alpha=0.25)
            axes[1].legend(loc="upper right", frameon=True)
        else:
            vals = np.asarray(validation_losses[:, 0], dtype=float)
            vals = np.where(vals > 0, vals, np.nan)
            axes[1].plot(epochs, vals, linewidth=1.8, label="total loss")
            axes[1].set_title(
                "total validation loss",
                loc="left",
                x=0.10,
                pad=8,
            )
            axes[1].set_xlabel("validation checkpoint")
            axes[1].set_ylabel("validation loss")
            axes[1].set_yscale("log")
            axes[1].grid(True, which="both", alpha=0.25)
            axes[1].legend(loc="upper right", frameon=True)

        # Both panels peak at their first sample, so the upper-left corner holds
        # the data maximum; the profile places the label lower-left instead.
        _cp().add_panel_labels(axes, profile="descending_line_panel")

        if save_dir is not None:
            save_path = save_dir / filename
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("saved training history plot to %s", save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)




def plot_reconstruction(
    test_data: Dict[str, np.ndarray],
    test_set_results: Dict[str, Any],
    dt: float,
    *,
    horizon_time: Optional[float] = None,
    horizon_crossed: bool = True,
    horizon_quantity: str = "reconstruction_validity_time",
    save_dir: Optional[str | Path] = None,
    show: bool = True,
    x_label: str = "Time",
    time_scale: float = 1.0,
    filename: str = "reconstruction.png",
    title_prefix: str = "a",
) -> float:
    time_grid = np.arange(test_data["x"].shape[0]) * dt * time_scale
    save_dir = _ensure_dir(save_dir)
    save_path = None if save_dir is None else save_dir / filename
    _plot_field_triptych_vertical(
        test_data["x"],
        test_set_results["x_decode"],
        time_grid,
        title_prefix=title_prefix,
        true_title=r"$u(x,t)$",
        pred_title=r"$\hat{u}(x,t)=\mathcal{D}_{\theta}(\mathcal{E}_{\phi}(u(x,t)))$",
        error_title=r"$u(x,t)-\hat{u}(x,t)$",
        true_colorbar_label=r"$u(x,t)$",
        pred_colorbar_label=r"$\hat{u}(x,t)$",
        error_colorbar_label=r"$u(x,t)-\hat{u}(x,t)$",
        horizon_time=horizon_time,
        horizon_crossed=horizon_crossed,
        horizon_quantity=horizon_quantity,
        save_path=save_path,
        show=show,
        x_label=x_label,
    )
    return float(test_set_results.get("decoder_error", np.nan))




def plot_xi_heatmap(
    test_set_results: Dict[str, Any],
    Xi: np.ndarray,
    term_names: Sequence[str],
    *,
    save_dir: Optional[str | Path] = None,
    show: bool = True,
    filename: str = "xi_heatmap.png",
    title_prefix: str = "f",
) -> None:
    """Plot the masked SINDy coefficient matrix."""
    save_dir = _ensure_dir(save_dir)

    Xi = np.asarray(Xi, dtype=float)
    active = int(np.count_nonzero(np.abs(Xi) > 0))
    total = int(Xi.size)

    vmax = float(np.nanmax(np.abs(Xi))) if np.any(np.isfinite(Xi)) and active > 0 else 1.0
    vmax = max(vmax, 1e-12)

    n_terms, latent_dim = Xi.shape
    latent_labels = [rf"$\dot{{z}}_{i + 1}$" for i in range(latent_dim)]

    # Type sizes come from the contract. The library-term axis carries many rows,
    # so its tick labels stay deliberately smaller than the contract default --
    # that difference is declared here rather than left as a bare literal.
    _rc = _ps().rc_params("ae_sindy_xi_heatmap")
    _term_label_pt = _rc["ytick.labelsize"] - 2.0
    with plt.rc_context({**_rc, "ytick.labelsize": _term_label_pt}):
        fig, ax = plt.subplots(
            1,
            1,
            figsize=_ps().figure_size_in("ae_sindy_xi_heatmap"),
            constrained_layout=True,
        )

        im = ax.imshow(
            Xi,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="none",
            # Without this the PDF embeds the raw 8 x 45 coefficient array and
            # the viewer smooths it when scaling to page size, which softens the
            # cell edges and washes out the colours -- and does so differently in
            # every viewer. Rasterising at figure dpi embeds the same crisp image
            # the PNG shows. Every other field artist in this module already does
            # this; the Xi heatmap was the one that did not.
            rasterized=True,
        )

        ax.set_title(
            r"learned SINDy coefficient matrix $\Xi$",
            loc="left",
            x=0.12,
            pad=8,
        )

        ax.set_xlabel(r"latent equation")
        ax.set_ylabel(r"library term in $\Theta(z)$")

        ax.set_xticks(np.arange(latent_dim))
        ax.set_xticklabels(latent_labels)

        ax.set_yticks(np.arange(n_terms))
        ax.set_yticklabels(term_names)

        ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True)
        ax.tick_params(axis="y", left=True, right=False)

        ax.set_xticks(np.arange(-0.5, latent_dim, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_terms, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.15, alpha=0.25)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.text(
            0.99,
            1.015,
            f"active = {active}/{total}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
        )

        # constrained_layout does not manage an axes created by
        # make_axes_locatable, so the colourbar and its label sat outside the
        # figure and bbox_inches="tight" silently grew the saved canvas from
        # 5.906 in to 6.603 in. Including that at the class width would have
        # shrunk every glyph by 10.6 %, which is the exact defect the printed-size
        # contract exists to prevent. fig.colorbar(..., ax=ax) is managed.
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
        cbar.set_label(r"coefficient value")

        if save_dir is not None:
            save_path = save_dir / filename
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("saved Xi heatmap to %s", save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)




def plot_latent_comparison(
    params: Dict[str, Any],
    test_set_results: Dict[str, Any],
    z_sim: np.ndarray,
    t_sim: np.ndarray,
    *,
    save_dir: Optional[str | Path] = None,
    show: bool = True,
    horizon_time: Optional[float] = None,
    horizon_crossed: bool = True,
    horizon_quantity: str = "reconstruction_validity_time",
    x_label: str = r"time $t$",
    filename: str = "latent_space_comparison.png",
    title_prefix: str = "b",
) -> None:
    """Plot encoder and SINDy latent trajectories."""
    save_dir = _ensure_dir(save_dir)

    n_sim = z_sim.shape[0]
    z_encoder = np.asarray(test_set_results["z"][:n_sim])
    z_sim = np.asarray(z_sim[:n_sim])
    t_sim = np.asarray(t_sim[:n_sim])

    latent_indices = np.arange(params["latent_dim"])
    latent_labels = [rf"$z_{{{i+1}}}$" for i in range(params["latent_dim"])]

    signed_residual = z_encoder - z_sim

    # Same repository-wide convention as the field triptychs: panels (a)/(b) share one
    # symmetric range derived from the encoder reference alone, panel (c) gets an
    # independent error range. The simulated latents previously widened the shared
    # range, which compressed the reference into an unreadable band whenever the
    # rollout diverged.
    vmin_z, vmax_z = _cp().truth_prediction_limits(z_encoder)
    err_vmin, err_vmax = _cp().error_limits(z_encoder, z_sim)

    with plt.rc_context(_ps().rc_params("ae_sindy_latent_comparison")):
        fig, axes = plt.subplots(
            3,
            1,
            figsize=_ps().figure_size_in("ae_sindy_latent_comparison"),
            sharex=True,
            constrained_layout=True,
        )

        # In-panel a/b/c labels; the mathematical latent
        # identifiers are scientifically meaningful and retained unchanged.
        panel_titles = [
            r"$z(t)=\mathcal{E}_{\phi}(u(x,t))$",
            r"$\hat{z}(t)$ from $\dot{z}=\Theta(z)\Xi$",
            r"$z(t)-\hat{z}(t)$",
        ]

        im0 = axes[0].pcolormesh(
            t_sim,
            latent_indices,
            z_encoder.T,
            cmap="viridis",
            shading="auto",
            vmin=vmin_z,
            vmax=vmax_z,
            rasterized=True,
        )
        axes[0].set_title(panel_titles[0], loc="left", x=0.35, pad=8)
        axes[0].set_ylabel(r"latent dimension $z$")
        axes[0].set_yticks(latent_indices)
        axes[0].set_yticklabels(latent_labels)
        # Colorbar carries no label: the panel identifier already names the field.
        fig.colorbar(im0, ax=axes[0], pad=0.025, fraction=0.045)

        im1 = axes[1].pcolormesh(
            t_sim,
            latent_indices,
            z_sim.T,
            cmap="viridis",
            shading="auto",
            vmin=vmin_z,
            vmax=vmax_z,
            rasterized=True,
        )
        axes[1].set_title(panel_titles[1], loc="left", x=0.35, pad=8)
        axes[1].set_ylabel(r"latent dimension $z$")
        axes[1].set_yticks(latent_indices)
        axes[1].set_yticklabels(latent_labels)
        fig.colorbar(im1, ax=axes[1], pad=0.025, fraction=0.045)

        im2 = axes[2].pcolormesh(
            t_sim,
            latent_indices,
            signed_residual.T,
            cmap="RdBu_r",
            shading="auto",
            vmin=err_vmin,
            vmax=err_vmax,
            rasterized=True,
        )
        axes[2].set_title(panel_titles[2], loc="left", x=0.35, pad=8)
        axes[2].set_xlabel(x_label)
        axes[2].set_ylabel(r"latent dimension $z$")
        axes[2].set_yticks(latent_indices)
        axes[2].set_yticklabels(latent_labels)
        fig.colorbar(im2, ax=axes[2], pad=0.025, fraction=0.045)

        for ax in axes:
            ax.grid(False)
            _add_horizon_marker(ax, horizon_time, horizon_crossed,
                                quantity=horizon_quantity)

        _cp().add_panel_labels(axes)

        if horizon_time is not None:
            axes[2].legend(loc="upper right", frameon=True)

        if save_dir is not None:
            save_path = save_dir / filename
            fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
            logger.info("saved latent comparison plot to %s", save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)




def plot_rollout_encoder_comparison(
    test_data: Dict[str, np.ndarray],
    test_set_results: Dict[str, Any],
    t_sim: np.ndarray,
    *,
    horizon_time: Optional[float] = None,
    horizon_crossed: bool = True,
    horizon_quantity: str = "reconstruction_validity_time",
    save_dir: Optional[str | Path] = None,
    show: bool = True,
    x_label: str = "Time",
    filename: str = "rollout_encoder_comparison.png",
    title_prefix: str = "c",
) -> None:
    """Plot the decoder output from encoder latents."""
    save_dir = _ensure_dir(save_dir)
    n_sim = len(t_sim)
    save_path = None if save_dir is None else save_dir / filename

    _plot_field_triptych_vertical(
        test_data["x"][:n_sim],
        test_set_results["x_decode"][:n_sim],
        t_sim,
        title_prefix=title_prefix,
        pred_title=r"$\hat{u}(x,t)=\mathcal{D}_{\theta}(\mathcal{E}_{\phi}(u(x,t)))$",
        error_title=r"$u(x,t)-\hat{u}(x,t)$",
        pred_colorbar_label=r"$\hat{u}(x,t)$",
        error_colorbar_label=r"$u(x,t)-\hat{u}(x,t)$",
        field_scale_mode="true_symmetric",
        error_scale_mode="field",
        horizon_time=horizon_time,
        horizon_crossed=horizon_crossed,
        horizon_quantity=horizon_quantity,
        save_path=save_path,
        show=show,
        x_label=x_label,
    )





def plot_rollout_sindy_comparison(
    test_data: Dict[str, np.ndarray],
    x_sindy: np.ndarray,
    t_sim: np.ndarray,
    *,
    horizon_time: Optional[float] = None,
    horizon_crossed: bool = True,
    horizon_quantity: str = "reconstruction_validity_time",
    save_dir: Optional[str | Path] = None,
    show: bool = True,
    x_label: str = "Time",
    filename: str = "rollout_sindy_comparison.png",
    title_prefix: str = "d",
) -> None:
    """Plot the decoded SINDy rollout in physical space."""
    save_dir = _ensure_dir(save_dir)
    n_sim = len(t_sim)
    x_true = np.asarray(test_data["x"][:n_sim])
    x_pred = np.asarray(x_sindy[:n_sim])
    signed_error = x_true - x_pred

    # Same repository-wide convention as every other triptych: panels (a)/(b) share a
    # symmetric range from the true field only, panel (c) gets its own error range.
    # Previously panel (c) reused the field limit, which hid errors larger than the
    # truth's amplitude.
    field_vmin, field_vmax = _cp().truth_prediction_limits(x_true)
    error_vmin, error_vmax = _cp().error_limits(x_true, x_pred)

    save_path = None if save_dir is None else save_dir / filename

    with plt.rc_context(_ps().rc_params("ae_sindy_field_triptych")):
        spatial_grid = _cp().kse_spatial_grid(x_true.shape[1])

        fig, axes = plt.subplots(
            3,
            1,
            figsize=_ps().figure_size_in("ae_sindy_field_triptych"),
            sharex=True,
            constrained_layout=True,
        )

        panel_titles = [
            r"$u(x,t)$",
            r"$\hat{u}(x,t)=\mathcal{D}_{\theta}(\hat{z}(t))$, $\hat{z}(t)$ from $\dot{z}=\Theta(z)\Xi$",
            r"$u(x,t)-\hat{u}(x,t)$",
        ]

        im0 = axes[0].pcolormesh(
            t_sim,
            spatial_grid,
            x_true.T,
            cmap="viridis",
            shading="auto",
            vmin=field_vmin,
            vmax=field_vmax,
            rasterized=True,
        )
        axes[0].set_title(panel_titles[0], loc="left", x=0.35, pad=8)
        axes[0].set_ylabel(r"space $x$")
        fig.colorbar(im0, ax=axes[0], pad=0.025, fraction=0.045)

        im1 = axes[1].pcolormesh(
            t_sim,
            spatial_grid,
            x_pred.T,
            cmap="viridis",
            shading="auto",
            vmin=field_vmin,
            vmax=field_vmax,
            rasterized=True,
        )
        axes[1].set_title(panel_titles[1], loc="left", x=0.35, pad=8)
        axes[1].set_ylabel(r"space $x$")
        fig.colorbar(im1, ax=axes[1], pad=0.025, fraction=0.045)

        im2 = axes[2].pcolormesh(
            t_sim,
            spatial_grid,
            signed_error.T,
            cmap="RdBu_r",
            shading="auto",
            vmin=error_vmin,
            vmax=error_vmax,
            rasterized=True,
        )
        axes[2].set_title(panel_titles[2], loc="left", x=0.35, pad=8)
        axes[2].set_xlabel(x_label)
        axes[2].set_ylabel(r"space $x$")
        fig.colorbar(im2, ax=axes[2], pad=0.025, fraction=0.045)

        for ax in axes:
            ax.set_ylim(spatial_grid[0], spatial_grid[-1])
            ax.grid(False)
            _add_horizon_marker(ax, horizon_time, horizon_crossed,
                                quantity=horizon_quantity)

        _cp().add_panel_labels(axes)

        if horizon_time is not None:
            axes[2].legend(loc="upper right", frameon=True)

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
            logger.info("saved plot to %s", save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

def visualize_run(
    run_dir: str | Path,
    *,
    data_file: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
    show: bool = False,
    config: Optional[AESINDyConfig] = None,
    time_axis_mode: Optional[str] = None,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_run_artifacts(run_dir)
    cfg = config if config is not None else load_config_from_run(run_dir)
    time_scale, x_label = _time_axis_from_config(cfg, time_axis_mode)
    test_data, _ = load_test_data_for_run(run_dir, data_file=data_file)

    validation_losses = artifacts.get("validation_losses")
    sindy_model_terms = artifacts.get("sindy_model_terms")
    if validation_losses is not None:
        plot_training_history(validation_losses, sindy_model_terms, save_dir=output_dir, show=show)

    summary = artifacts.get("analysis_summary")
    if summary is None:
        raise FileNotFoundError(f"analysis_summary.pkl not found in {run_dir}. Run analysis first.")

    results = summary["results"]
    Xi = summary["Xi"]
    term_names = summary["term_names"]
    plot_reconstruction(
        test_data,
        results,
        cfg.dt,
        horizon_time=_scale_horizon_time(summary.get("encoder_valid_time"), time_scale),
        horizon_crossed=bool(summary.get("encoder_threshold_crossed", True)),
        horizon_quantity="reconstruction_validity_time",
        save_dir=output_dir,
        show=show,
        x_label=x_label,
        time_scale=time_scale,
    )
    plot_xi_heatmap(results, Xi, term_names, save_dir=output_dir, show=show)

    if summary.get("z_sim") is not None and summary.get("t_sim") is not None:
        plot_latent_comparison(
            cfg.to_params_dict(run_dir=run_dir),
            results,
            summary["z_sim"],
            np.asarray(summary["t_sim"]) * time_scale,
            horizon_time=_scale_horizon_time(summary.get("sindy_valid_time"), time_scale),
            horizon_quantity="prediction_horizon",
            horizon_crossed=bool(summary.get("sindy_threshold_crossed", True)),
            save_dir=output_dir,
            show=show,
            x_label=x_label,
        )
        plot_rollout_encoder_comparison(
            test_data,
            results,
            np.asarray(summary["t_sim"]) * time_scale,
            horizon_time=_scale_horizon_time(summary.get("encoder_rollout_valid_time"), time_scale),
            horizon_crossed=bool(summary.get("encoder_rollout_threshold_crossed", True)),
            horizon_quantity="reconstruction_validity_time",
            save_dir=output_dir,
            show=show,
            x_label=x_label,
        )
    if summary.get("x_sindy") is not None and summary.get("t_sim") is not None:
        plot_rollout_sindy_comparison(
            test_data,
            summary["x_sindy"],
            np.asarray(summary["t_sim"]) * time_scale,
            horizon_time=_scale_horizon_time(summary.get("sindy_valid_time"), time_scale),
            horizon_quantity="prediction_horizon",
            horizon_crossed=bool(summary.get("sindy_threshold_crossed", True)),
            save_dir=output_dir,
            show=show,
            x_label=x_label,
        )

    logger.info("saved visualization figures to %s", output_dir)
    return {"output_dir": str(output_dir)}
