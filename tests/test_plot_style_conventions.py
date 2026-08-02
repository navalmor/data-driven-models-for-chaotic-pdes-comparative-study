"""Guard tests for the central plotting style contract (Phase 2, CH-18).

These lock the conventions that previously drifted without anyone noticing:

  * the horizon symbol silently disagreed with the axis it was plotted on for
    every forecasting figure in the thesis;
  * a figure drawn four times larger than it is printed had its type sizes
    reduced to 4.5 pt on paper;
  * a centred label on the outermost tick ran into the neighbouring panel;
  * two annotations placed by a fixed guess crossed their axes frame.

Every one of those is a geometric or textual property that can be asserted from
a rendered figure, so each has a test here rather than a note in a document.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (str(REPO_ROOT), str(REPO_ROOT / "forecasting/src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from common import plot_style as PS  # noqa: E402

OKABE_ITO = {"#000000", "#E69F00", "#56B4E9", "#009E73",
             "#F0E442", "#0072B2", "#D55E00", "#CC79A7"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _write_run(root: Path, split: str, *, metrics: dict, config: dict | None = None,
               n_time: int = 40, n_space: int = 8) -> Path:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(n_time, n_space))
    np.save(split_dir / "truth_physical.npy", truth)
    np.save(split_dir / "pred_physical.npy", truth + 0.25)
    # the error series the horizon plot reads
    times = np.arange(n_time) * float(metrics.get("dt", 0.1))
    rel = np.linspace(0.02, 1.4, n_time)
    with (split_dir / "errors_physical.csv").open("w") as fh:
        fh.write("time,relative_l2\n")
        for t, r in zip(times, rel):
            fh.write(f"{t},{r}\n")

    (split_dir / "metrics_physical.json").write_text(json.dumps(metrics))
    if config is not None:
        (root / "config_resolved.json").write_text(json.dumps(config))
    return root


def _render(fn, **kwargs):
    """Run a plotting function and hand back the figure before it is saved."""
    from unittest import mock
    from forecasting.visualization import forecast_plots as FP

    captured: dict = {}

    def _capture(fig, output_base, formats, dpi, **kw):
        captured["fig"] = fig
        return []

    with mock.patch.object(FP, "save_figure", _capture):
        fn(**kwargs)
    fig = captured["fig"]
    fig.canvas.draw()
    return fig


def _forecast_kwargs(tmp: Path, *, cutoff: bool, split: str = "validation"):
    metrics = {"dt": 0.1, "relative_l2_threshold": 0.5,
               "relative_l2_horizon_time": 1.2}
    if cutoff:
        metrics["prediction_masked_from_index"] = 20
    root = _write_run(tmp, split, metrics=metrics)
    return dict(run_dir=root, split=split, output_base=tmp / "out",
                formats=["png"], dpi=50, use_lyapunov=True, lambda_max=0.043)


def _all_text_artists(ax):
    """Every text artist an axes owns, including its three title slots.

    ``ax.texts`` holds only free annotations. A title set with ``loc="left"``
    lives in ``ax._left_title``, not ``ax.title`` -- checking only ``ax.texts``
    is how a 13 pt left title that overflowed its panel by 65 px and landed on
    the neighbouring y-axis label went unnoticed.
    """
    out = list(ax.texts)
    for name in ("title", "_left_title", "_right_title"):
        t = getattr(ax, name, None)
        if t is not None:
            out.append(t)
    out += [ax.xaxis.label, ax.yaxis.label]
    return [t for t in out if t.get_text().strip()]


def _visible_xticklabels(ax):
    """Tick labels matplotlib will actually draw, i.e. inside the x limits.

    ``get_xticklabels`` also returns labels for tick locations outside the view;
    those are never rendered but still report a window extent, so including them
    invents overlaps that do not exist on the page.
    """
    lo, hi = ax.get_xlim()
    out = []
    for loc, lab in zip(ax.get_xticks(), ax.get_xticklabels()):
        if lo - 1e-9 <= loc <= hi + 1e-9 and lab.get_text().strip():
            out.append(lab)
    return out


def _texts_of(ax):
    out = list(ax.texts)
    out += [ax.xaxis.label, ax.yaxis.label, ax.title]
    out += ax.get_xticklabels() + ax.get_yticklabels()
    return [t for t in out if t.get_text().strip()]


# ---------------------------------------------------------------------------
class TestStyleContractIntegrity(unittest.TestCase):
    """The contract itself must be complete and internally consistent."""

    def test_every_family_resolves_to_a_printed_size(self):
        for name in PS.families():
            derived = PS.family(name)["aspect_ratio"] == "derived"
            w, h = PS.figure_size_in(name, aspect_ratio=0.9 if derived else None)
            self.assertGreater(w, 0, name)
            self.assertGreater(h, 0, name)
            self.assertLessEqual(w, PS.style()["page"]["text_height_in"] + 1e-6, name)

    def test_pdf_fonttype_is_truetype(self):
        # Type 3 loses the lambda glyph on text extraction and is poorly
        # regarded for print; 42 embeds TrueType and is also smaller here.
        for name in PS.families():
            self.assertEqual(PS.rc_params(name)["pdf.fonttype"], 42, name)

    def test_pdf_classification_is_declared_and_sampling_matrix_is_hybrid(self):
        for name in PS.families():
            self.assertIn(PS.pdf_class(name), {"vector", "hybrid"}, name)
        # The scatter and histograms are vector, but the colourbar gradient
        # remains one raster, so the matrix is hybrid rather than fully vector.
        self.assertEqual(PS.pdf_class("optimizer_sampling_matrix"), "hybrid")

    def test_palette_is_okabe_ito(self):
        for key, value in PS.palette().items():
            if key == "grey":
                continue
            self.assertIn(value.upper(), OKABE_ITO, f"{key}={value}")

    def test_marker_styles_are_distinct_and_declared(self):
        hor, cut = PS.marker("prediction_horizon"), PS.marker("safety_cutoff")
        self.assertEqual(hor["linestyle"], "--")
        self.assertEqual(cut["linestyle"], ":")
        self.assertEqual(hor["color"], cut["color"])   # same colour, different dash
        self.assertNotEqual(hor["linestyle"], cut["linestyle"])

    def test_panel_label_policy_sets_are_unchanged(self):
        # The policy is to PRESERVE what each family already does. This test
        # exists so that "add labels everywhere for uniformity" cannot happen
        # by accident.
        policy = PS.panel_label_policy()
        # Verified against the add_panel_labels call sites in the plotting
        # sources, not from a visual reading: ae_sindy_training_history has
        # carried boxed a/b labels all along.
        self.assertEqual(set(policy["boxed_families"]),
                         {"spatiotemporal_comparison", "ae_sindy_field_triptych",
                          "ae_sindy_latent_comparison", "ae_sindy_training_history"})
        self.assertEqual(set(policy["title_embedded_families"]), {"physics_ablation"})
        every = set(policy["boxed_families"]) | set(policy["title_embedded_families"]) \
            | set(policy["unlabelled_families"])
        self.assertEqual(every, set(PS.families()))


class TestNotation(unittest.TestCase):
    """T_h is physical time; t_lambda is a Lyapunov-axis position. Never mixed."""

    def test_symbols_follow_the_axis(self):
        self.assertEqual(PS.horizon_symbol(True), r"t_\lambda")
        self.assertEqual(PS.horizon_symbol(False), "T_h")
        self.assertEqual(PS.cutoff_symbol(True), r"t_{\lambda,c}")
        self.assertEqual(PS.cutoff_symbol(False), "t_c")

    def test_lyapunov_labels_never_say_T_h(self):
        for label in (PS.horizon_label(1.05, True), PS.cutoff_label(2.53, True)):
            self.assertNotIn("T_h", label)
            self.assertIn(r"\lambda", label)

    def test_physical_labels_never_say_t_lambda(self):
        for label in (PS.horizon_label(24.4, False), PS.cutoff_label(2.53, False)):
            self.assertNotIn(r"t_\lambda", label)
            self.assertIn("T_h" if "horizon" in label.lower() else "t_c", label)

    def test_T_lambda_is_never_produced(self):
        # T_lambda was explicitly rejected: T is the physical-time symbol.
        for label in (PS.horizon_label(1.0, True), PS.horizon_label(1.0, False),
                      PS.cutoff_label(1.0, True), PS.cutoff_label(1.0, False)):
            self.assertNotIn(r"T_\lambda", label)


class TestNoDownscaling(unittest.TestCase):
    """A declared type size is only real if the figure is drawn at printed size."""

    def test_every_family_is_drawn_at_its_printed_width(self):
        for name in PS.families():
            derived = PS.family(name)["aspect_ratio"] == "derived"
            drawn_w, _ = PS.figure_size_in(name, aspect_ratio=0.9 if derived else None)
            self.assertAlmostEqual(PS.downscale_factor(name, drawn_w), 1.0, places=6,
                                   msg=name)

    def test_historical_oversizing_is_detected(self):
        # The regression this test exists for: 16.79 in drawn, 8.90 in printed.
        factor = PS.downscale_factor("optimizer_sampling_matrix", 16.79)
        self.assertLess(factor, 0.6)
        self.assertLess(PS.effective_printed_pt("optimizer_sampling_matrix", 10.0, 16.79),
                        6.0)

    def test_declared_type_sizes_clear_the_legibility_floor(self):
        floor = PS.minimum_printed_pt()
        for name in PS.families():
            rc = PS.rc_params(name)
            self.assertGreaterEqual(rc["xtick.labelsize"], floor["tick_label"], name)
            self.assertGreaterEqual(rc["axes.labelsize"], floor["axis_label"], name)


class TestRenderedGeometry(unittest.TestCase):
    """Geometric properties that can only be checked on a drawn figure."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _error_curve(self, cutoff=True):
        from forecasting.visualization.forecast_plots import plot_relative_l2_horizon
        return _render(plot_relative_l2_horizon,
                       **_forecast_kwargs(self.root, cutoff=cutoff))

    def _triptych(self):
        from forecasting.visualization.forecast_plots import plot_spatiotemporal_comparison
        kw = _forecast_kwargs(self.root, cutoff=True)
        kw.update(model_name="RC", representation="full64", max_time_steps=None)
        return _render(plot_spatiotemporal_comparison, **kw)

    # ---- text must stay inside its axes ------------------------------------
    def test_no_text_crosses_its_axes_frame(self):
        for fig in (self._error_curve(), self._triptych()):
            renderer = fig.canvas.get_renderer()
            for ax in fig.axes:
                frame = ax.get_window_extent(renderer)
                for text in ax.texts:
                    bb = text.get_window_extent(renderer)
                    self.assertTrue(
                        bb.x0 >= frame.x0 - 1 and bb.x1 <= frame.x1 + 1,
                        f"annotation {text.get_text()!r} crosses its axes frame")
            plt.close(fig)

    def _sampling_matrix(self):
        """The only family with categorical tick labels in adjacent panels.

        This is rendered from the canonical tracked trial data rather than from
        synthetic rows, because the defect it guards against depends on the real
        category names -- "Linear + quadratic" beside "diagonal" is what collided.
        """
        from forecasting.visualization import search_plots as SP
        from forecasting.visualization.io import load_optimizer_results

        cfg = json.loads((REPO_ROOT / "final_thesis_package_v001/configs/03_plots"
                          / "optimizer_search_ngrc.json").read_text())
        s_cfg = cfg["search_plots"]
        group = [g for g in cfg["inputs"]["optimizer_groups"]
                 if g["label"] == "ngrc_latent8"][0]
        rows = SP.ok_rows(
            load_optimizer_results([REPO_ROOT / d for d in group["search_dirs"]]),
            s_cfg["metric"])
        specs = SP.build_parameter_specs(s_cfg["parameters"])

        from unittest import mock
        got = {}

        def _cap(fig, output_base, formats, dpi, **kw):
            got["fig"] = fig
            return []

        with mock.patch.object(SP, "save_figure", _cap):
            SP.plot_sampling_matrix(
                rows, specs=specs, metric=s_cfg["metric"],
                objective=s_cfg["objective"], output_base=self.root / "m",
                formats=["png"], dpi=100,
                use_lyapunov=bool(s_cfg.get("use_lyapunov", False)),
                lambda_max=float(s_cfg.get("lambda_max", 1.0)))
        fig = got["fig"]
        fig.canvas.draw()
        return fig

    # ---- tick labels must not reach into a neighbouring axes ---------------
    def test_categorical_tick_labels_do_not_overlap_across_panels(self):
        """The regression this exists for: "Linear +" ran into "diagonal".

        A tick label is centred on its tick, so the OUTERMOST label of a panel
        overhangs by half its own width. At printed cell width that overhang
        reaches the neighbouring panel. Only the sampling matrix has adjacent
        panels with categorical labels, so only the sampling matrix can catch it.
        """
        fig = self._sampling_matrix()
        renderer = fig.canvas.get_renderer()
        boxes = []
        for i, ax in enumerate(fig.axes):
            for lab in _visible_xticklabels(ax):
                boxes.append((i, lab.get_text(), lab.get_window_extent(renderer)))
        self.assertGreater(len(boxes), 4, "no categorical tick labels were rendered")
        for i, (ai, ta, ba) in enumerate(boxes):
            for aj, tb, bb in boxes[i + 1:]:
                if ai == aj:
                    continue
                overlap = (ba.x0 < bb.x1 and bb.x0 < ba.x1
                           and ba.y0 < bb.y1 and bb.y0 < ba.y1)
                self.assertFalse(overlap,
                                 f"tick labels {ta!r} and {tb!r} overlap across panels")
        plt.close(fig)

    def test_outermost_categorical_labels_stay_inside_their_panel(self):
        """The mechanism, asserted directly rather than only via its symptom.

        Scoped to CATEGORICAL labels. A numeric tick label centred on the first
        or last tick overhangs its axes slightly, which is ordinary matplotlib
        behaviour and harmless -- anchoring it inward would misalign it from the
        tick it names, which is worse. Wide multi-word category names are the
        ones that overhang far enough to reach the next panel, and those are the
        ones the anchoring applies to. Numeric overhang is governed instead by
        the collision test above, which is the property that actually matters.
        """
        fig = self._sampling_matrix()
        renderer = fig.canvas.get_renderer()
        checked = 0
        for i, ax in enumerate(fig.axes):
            labels = _visible_xticklabels(ax)
            if len(labels) < 2:
                continue
            categorical = [l for l in labels
                           if not re.fullmatch(r"[-−]?[\d.]+",
                                               l.get_text().strip())]
            if len(categorical) < 2:
                continue
            frame = ax.get_window_extent(renderer)
            for lab in (categorical[0], categorical[-1]):
                checked += 1
                bb = lab.get_window_extent(renderer)
                self.assertTrue(
                    bb.x0 >= frame.x0 - 1 and bb.x1 <= frame.x1 + 1,
                    f"axes[{i}] outer category {lab.get_text()!r} leaves its panel")
        self.assertGreater(checked, 0, "no categorical outer labels were checked")
        plt.close(fig)

    def test_triptych_tick_labels_do_not_overlap_across_axes(self):
        fig = self._triptych()
        renderer = fig.canvas.get_renderer()
        boxes = []
        for i, ax in enumerate(fig.axes):
            for lab in _visible_xticklabels(ax):
                boxes.append((i, lab.get_text(), lab.get_window_extent(renderer)))
        for i, (ai, ta, ba) in enumerate(boxes):
            for aj, tb, bb in boxes[i + 1:]:
                if ai == aj:
                    continue
                overlap = (ba.x0 < bb.x1 and bb.x0 < ba.x1
                           and ba.y0 < bb.y1 and bb.y0 < ba.y1)
                self.assertFalse(overlap,
                                 f"tick labels {ta!r} and {tb!r} overlap across axes")
        plt.close(fig)

    # ---- the legend must not cover the markers it explains -----------------
    def test_no_title_or_label_overlaps_a_neighbouring_axes(self):
        """Titles and axis labels, not just annotations.

        A left-anchored title wider than its own panel reaches into the panel
        beside it. This is checked across every pair of axes, on the multi-panel
        families where two panels sit side by side.
        """
        for fig in (self._triptych(), self._sampling_matrix()):
            renderer = fig.canvas.get_renderer()
            items = []
            for i, ax in enumerate(fig.axes):
                for t in _all_text_artists(ax):
                    items.append((i, t.get_text()[:40], t.get_window_extent(renderer)))
            for i, (ai, ta, ba) in enumerate(items):
                for aj, tb, bb in items[i + 1:]:
                    if ai == aj:
                        continue
                    overlap = (ba.x0 < bb.x1 and bb.x0 < ba.x1
                               and ba.y0 < bb.y1 and bb.y0 < ba.y1)
                    self.assertFalse(
                        overlap,
                        f"text {ta!r} overlaps {tb!r} from a different axes")
            plt.close(fig)

    def test_titles_stay_inside_their_own_axes(self):
        """A title must not be wider than the panel it names."""
        for fig in (self._triptych(), self._sampling_matrix()):
            renderer = fig.canvas.get_renderer()
            for i, ax in enumerate(fig.axes):
                frame = ax.get_window_extent(renderer)
                for name in ("title", "_left_title", "_right_title"):
                    t = getattr(ax, name, None)
                    if t is None or not t.get_text().strip():
                        continue
                    bb = t.get_window_extent(renderer)
                    self.assertLessEqual(
                        bb.x1, frame.x1 + 1,
                        f"axes[{i}] {name} {t.get_text()[:40]!r} overflows its panel "
                        f"by {bb.x1 - frame.x1:.1f} px")
            plt.close(fig)

    def test_legend_does_not_cover_horizon_or_cutoff_markers(self):
        fig = self._error_curve()
        renderer = fig.canvas.get_renderer()
        ax = fig.axes[0]
        legend = ax.get_legend()
        self.assertIsNotNone(legend, "the error curve must carry a legend")
        lb = legend.get_window_extent(renderer)
        for line in ax.get_lines():
            xd = np.asarray(line.get_xdata(), dtype=float)
            if xd.size != 2 or not np.allclose(xd[0], xd[1]):
                continue                                    # not a vertical marker
            x_px, _ = ax.transData.transform((xd[0], ax.get_ylim()[0]))
            self.assertFalse(lb.x0 <= x_px <= lb.x1,
                             f"legend covers marker {line.get_label()!r}")
        plt.close(fig)

    def test_legend_placement_is_explicit_not_best(self):
        # Comments may discuss loc="best"; only a live call is a defect.
        from forecasting.visualization import forecast_plots as FP

        code = re.sub(r"(?m)#.*$", "", Path(FP.__file__).read_text())
        calls = re.findall(r"\.legend\([^)]*loc\s*=\s*[\"\']best[\"\']", code)
        self.assertEqual(calls, [],
                         'loc="best" moves per case and can land on the data')


class TestRenderedConventions(unittest.TestCase):
    """Textual conventions, asserted on what a figure actually prints."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _strings(self, fig):
        out = []
        for ax in fig.axes:
            out += [t.get_text() for t in _texts_of(ax)]
            leg = ax.get_legend()
            if leg:
                out += [t.get_text() for t in leg.get_texts()]
        out += [t.get_text() for t in fig.texts]
        return [s for s in out if s.strip()]

    def test_lyapunov_figure_uses_t_lambda_and_never_T_h(self):
        from forecasting.visualization.forecast_plots import plot_relative_l2_horizon
        fig = _render(plot_relative_l2_horizon,
                      **_forecast_kwargs(self.root, cutoff=True))
        strings = self._strings(fig)
        self.assertTrue(any(r"t_\lambda" in s for s in strings), strings)
        self.assertFalse(any("T_h" in s for s in strings), strings)
        plt.close(fig)

    def test_retired_phrases_are_absent(self):
        from forecasting.visualization.forecast_plots import plot_relative_l2_horizon
        fig = _render(plot_relative_l2_horizon,
                      **_forecast_kwargs(self.root, cutoff=True))
        strings = " | ".join(self._strings(fig))
        for retired in PS.retired_terms():
            self.assertNotIn(retired, strings)
        plt.close(fig)

    def test_axis_labels_are_lowercase(self):
        from forecasting.visualization.forecast_plots import plot_relative_l2_horizon
        fig = _render(plot_relative_l2_horizon,
                      **_forecast_kwargs(self.root, cutoff=False))
        for ax in fig.axes:
            for label in (ax.get_xlabel(), ax.get_ylabel()):
                if label and label[0].isalpha():
                    self.assertTrue(label[0].islower(),
                                    f"axis label {label!r} should be lowercase")
        plt.close(fig)

    def test_cutoff_gets_its_own_line_not_only_shading(self):
        from forecasting.visualization.forecast_plots import plot_relative_l2_horizon
        fig = _render(plot_relative_l2_horizon,
                      **_forecast_kwargs(self.root, cutoff=True))
        labels = [l.get_label() for l in fig.axes[0].get_lines()]
        self.assertTrue(any("Safety cutoff" in str(l) for l in labels), labels)
        plt.close(fig)


class TestSourceHygiene(unittest.TestCase):
    """Sizes, colours and type scales belong to the contract, not to call sites.

    The guard previously covered two of the six modules that actually draw, and
    tested only figsize and hex colours. A hard-coded ``fontsize=`` therefore
    passed even in a listed module, which is how pre-contract type sizes survived
    unnoticed. All six drawing modules are now covered, and the checks include the
    type-size keywords and locally-built Matplotlib style dictionaries.
    """

    #: Every module in the repository that creates a Matplotlib figure. Kept
    #: explicit rather than globbed, so adding a new drawing module is a
    #: deliberate act that also brings it under this guard.
    MODULES = [
        "ae_sindy/ae_sindy/visualization.py",
        "forecasting/scripts/plot_physics_ablation.py",
        "forecasting/scripts/plot_seed_robustness.py",
        "forecasting/src/forecasting/visualization/forecast_plots.py",
        "forecasting/src/forecasting/visualization/search_plots.py",
        "simulation/kse_simulation.py",
    ]

    #: rcParams keys that fix a printed size. A literal value for any of these
    #: outside common/plot_style.json silently overrides the contract.
    STYLE_KEYS = (
        "font.size", "axes.titlesize", "axes.labelsize", "legend.fontsize",
        "xtick.labelsize", "ytick.labelsize", "figure.titlesize",
        "lines.linewidth", "axes.linewidth",
    )

    #: Directory names that never hold repository source. ``site-packages``
    #: carries installed third-party code, which draws figures constantly.
    NON_SOURCE_DIRS = frozenset({"site-packages", "__pycache__"})

    def _sources(self):
        for rel in self.MODULES:
            path = REPO_ROOT / rel
            self.assertTrue(path.is_file(), f"declared drawing module missing: {rel}")
            yield rel, path.read_text()

    @staticmethod
    def _virtualenv_roots(root: Path) -> set[Path]:
        """Directories that are virtual environments, found by their marker file.

        README documents ``python3.12 -m venv .venv`` inside the clone, but the
        name is the reader's choice, so ``pyvenv.cfg`` rather than the directory
        name is what identifies an environment.
        """
        return {cfg.parent for cfg in root.rglob("pyvenv.cfg")}

    @classmethod
    def _modules_that_draw(cls, root: Path) -> set[str]:
        """Repository Python files that create a Matplotlib figure.

        Installed packages inside an in-repository virtual environment are not
        repository source and are skipped before their contents are read.
        """
        venvs = cls._virtualenv_roots(root)
        drawing = set()
        for path in root.rglob("*.py"):
            rel = path.relative_to(root).as_posix()
            if rel.startswith(("tests/", "_local_review/", "outputs/",
                               "reproduction_runs/", "common/plot_style")):
                continue
            if cls.NON_SOURCE_DIRS.intersection(path.parts):
                continue
            if any(venv in path.parents for venv in venvs):
                continue
            src = path.read_text(errors="ignore")
            if "plt.subplots(" in src or "plt.figure(" in src:
                drawing.add(rel)
        return drawing

    def test_the_module_list_covers_every_module_that_draws(self):
        """A new drawing module must not escape this guard by not being listed."""
        drawing = self._modules_that_draw(REPO_ROOT)
        unguarded = sorted(drawing - set(self.MODULES))
        self.assertEqual(unguarded, [],
                         f"modules create figures but are not in MODULES: {unguarded}")
        # the exclusions must not have hidden real source: all six are still found
        self.assertEqual(sorted(drawing), sorted(self.MODULES))

    def _fake_repo(self, tmp: str) -> Path:
        """A miniature repository holding real source and two environments."""
        root = Path(tmp)
        (root / "forecasting/scripts").mkdir(parents=True)
        (root / "forecasting/scripts/plot_new_thing.py").write_text(
            "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\n")
        # the documented name, and a differently named environment, so the test
        # fails if detection ever falls back to matching ".venv"
        for name in (".venv", "env312"):
            site = root / name / "lib/python3.12/site-packages/thirdparty"
            site.mkdir(parents=True)
            (root / name / "pyvenv.cfg").write_text("home = /usr/bin\n")
            (site / "renderer.py").write_text(
                "import matplotlib.pyplot as plt\nplt.figure()\n")
        return root

    def test_a_virtual_environment_inside_the_repository_is_not_scanned(self):
        """The documented setup puts .venv in the clone; it is not our source."""
        with tempfile.TemporaryDirectory() as tmp:
            drawing = self._modules_that_draw(self._fake_repo(tmp))
        self.assertNotIn(".venv/lib/python3.12/site-packages/thirdparty/renderer.py",
                         drawing)
        self.assertNotIn("env312/lib/python3.12/site-packages/thirdparty/renderer.py",
                         drawing)

    def test_an_unlisted_drawing_module_is_still_detected(self):
        """Excluding environments must not blind the guard to repository source."""
        with tempfile.TemporaryDirectory() as tmp:
            drawing = self._modules_that_draw(self._fake_repo(tmp))
        self.assertEqual(drawing, {"forecasting/scripts/plot_new_thing.py"})

    def test_no_hard_coded_figsize_outside_the_contract(self):
        for rel, src in self._sources():
            literal = re.findall(r"figsize=\(\s*[\d.]+\s*,", src)
            self.assertEqual(literal, [],
                             f"{rel} sets a literal figsize: {literal}")

    def test_no_hard_coded_hex_colour_outside_the_contract(self):
        for rel, src in self._sources():
            hexes = {h.upper() for h in re.findall(r'"(#[0-9a-fA-F]{6})"', src)}
            self.assertEqual(hexes, set(),
                             f"{rel} hard-codes colours {hexes}; use plot_style")

    def test_no_literal_fontsize_or_labelsize(self):
        """A literal type size wins over the rc context that encloses it."""
        for rel, src in self._sources():
            literal = re.findall(r"\b(?:fontsize|labelsize|titlesize)\s*=\s*[\d.]+", src)
            self.assertEqual(literal, [],
                             f"{rel} sets a literal type size {literal}; "
                             "take it from plot_style.rc_params(family)")

    def test_no_local_matplotlib_style_dictionary(self):
        """A literal rc dict bypasses the contract as completely as a fontsize does."""
        for rel, src in self._sources():
            offenders = [
                key for key in self.STYLE_KEYS
                if re.search(r'["\']' + re.escape(key) + r'["\']\s*:\s*[\d.]+', src)
            ]
            self.assertEqual(offenders, [],
                             f"{rel} builds a local style dict setting {offenders}; "
                             "use plot_style.rc_params(family)")

    def test_every_drawing_module_reads_the_central_contract(self):
        for rel, src in self._sources():
            self.assertIn("plot_style", src,
                          f"{rel} draws figures without reading common/plot_style")


class TestOptimizerConfigsPointAtTrackedData(unittest.TestCase):
    """The published optimizer plots must be rebuildable from their own config."""

    def test_every_search_dir_exists_in_this_repository(self):
        configs = sorted((REPO_ROOT / "final_thesis_package_v001/configs/03_plots")
                         .glob("optimizer_search_*.json"))
        self.assertEqual(len(configs), 4)
        for cfg in configs:
            data = json.loads(cfg.read_text())
            for group in data["inputs"]["optimizer_groups"]:
                for search_dir in group["search_dirs"]:
                    self.assertTrue((REPO_ROOT / search_dir).is_dir(),
                                    f"{cfg.name}: {search_dir} is not in the repository")

    def test_original_experiment_paths_are_preserved_as_provenance(self):
        cfg = (REPO_ROOT / "final_thesis_package_v001/configs/03_plots"
               / "optimizer_search_ngrc.json")
        for group in json.loads(cfg.read_text())["inputs"]["optimizer_groups"]:
            self.assertIn("search_dirs_original_experiment_paths", group)


if __name__ == "__main__":
    unittest.main(verbosity=2)
