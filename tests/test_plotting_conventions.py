"""Plot-convention policy tests (standard library only).

These tests exercise the pure plotting helpers and statically inspect the
plotting source so that publication conventions cannot silently regress. They
load no dataset, train no model, run no optimiser search, and generate no heavy
figures.
"""

from __future__ import annotations

import ast
import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "forecasting" / "src"), str(_REPO_ROOT / "ae_sindy")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import plotting as P  # noqa: E402


def _read(path: Path) -> str:
    """Read a source file for static (AST/text) convention checks."""
    return path.read_text(encoding="utf-8")


class TestTimeAxisResolution(unittest.TestCase):
    def test_default_is_lyapunov(self):
        self.assertEqual(P.resolve_time_axis_mode(), "lyapunov")

    def test_physical_via_cli_and_config(self):
        self.assertEqual(P.resolve_time_axis_mode(cli_value="physical"), "physical")
        self.assertEqual(P.resolve_time_axis_mode(config_value="physical"), "physical")

    def test_cli_outranks_config(self):
        self.assertEqual(
            P.resolve_time_axis_mode(cli_value="physical", config_value="lyapunov"), "physical"
        )
        self.assertEqual(
            P.resolve_time_axis_mode(cli_value="lyapunov", config_value="physical"), "lyapunov"
        )

    def test_config_outranks_default(self):
        self.assertEqual(P.resolve_time_axis_mode(config_value="physical"), "physical")

    def test_invalid_mode_raises(self):
        for bad in ("lyap", "", "PHYSICALX", "none"):
            with self.assertRaises(ValueError):
                P.resolve_time_axis_mode(cli_value=bad)

    def test_resolve_from_args_and_config_dict(self):
        class Args:
            time_axis_mode = None

        self.assertEqual(P.resolve_time_axis_mode_from(Args(), {"plotting": {"time_axis_mode": "physical"}}), "physical")
        Args.time_axis_mode = "lyapunov"
        self.assertEqual(P.resolve_time_axis_mode_from(Args(), {"plotting": {"time_axis_mode": "physical"}}), "lyapunov")
        # missing plotting section falls through to default
        self.assertEqual(P.resolve_time_axis_mode_from(None, {}), "lyapunov")


class TestTimeAxisConversion(unittest.TestCase):
    def test_labels(self):
        _, lab_l = P.time_axis_scale_and_label("lyapunov", dt=0.1, lambda_max=0.043)
        _, lab_p = P.time_axis_scale_and_label("physical", dt=0.1)
        self.assertEqual(lab_l, r"time $t$ [$1/\lambda_{\max}$]")
        self.assertEqual(lab_p, r"time $t$")

    def test_scale_values(self):
        s_l, _ = P.time_axis_scale_and_label("lyapunov", lambda_max=0.043)
        s_p, _ = P.time_axis_scale_and_label("physical")
        self.assertEqual(s_l, 0.043)
        self.assertEqual(s_p, 1.0)

    def test_scale_time_no_mutation(self):
        a = np.arange(5.0)
        a_copy = a.copy()
        b = P.scale_time(a, 0.043)
        self.assertIsNot(b, a)
        self.assertTrue(np.allclose(a, a_copy))
        self.assertTrue(np.allclose(b, a_copy * 0.043))

    def test_sample_count_preserved_and_endpoints(self):
        idx = np.arange(1000) * 0.1  # 1000-step rollout, dt=0.1
        s_p, _ = P.time_axis_scale_and_label("physical")
        s_l, _ = P.time_axis_scale_and_label("lyapunov", lambda_max=0.043)
        tp = P.scale_time(idx, s_p)
        tl = P.scale_time(idx, s_l)
        self.assertEqual(len(tp), len(tl), 1000)
        self.assertAlmostEqual(tp[-1], 99.9, places=6)
        self.assertAlmostEqual(tl[-1], 4.2957, places=3)
        self.assertTrue(np.allclose(tl, tp * 0.043))

    def test_positive_value_validation(self):
        with self.assertRaises(ValueError):
            P.time_axis_scale_and_label("lyapunov", lambda_max=0.0)
        with self.assertRaises(ValueError):
            P.time_axis_scale_and_label("physical", dt=-1.0)

    def test_no_randomness_in_module(self):
        # AST-based so docstring prose ("consumes no randomness") is not counted:
        # assert the module references no `random` attribute or name in code.
        src = (_REPO_ROOT / "common" / "plotting.py").read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"random", "default_rng", "seed", "shuffle"}:
                self.fail("common.plotting references a randomness API")
            if isinstance(node, ast.Name) and node.id == "random":
                self.fail("common.plotting references the random module")


class TestPanelLabels(unittest.TestCase):
    def test_sequence(self):
        self.assertEqual([P.panel_label(i) for i in range(3)], ["(a)", "(b)", "(c)"])

    def test_panel_title_attached(self):
        self.assertEqual(P.panel_title(0, "$u(x,t)$"), "(a) $u(x,t)$")
        self.assertEqual(P.panel_title(1, "active SINDy terms"), "(b) active SINDy terms")

    def test_out_of_range(self):
        with self.assertRaises(ValueError):
            P.panel_label(-1)
        with self.assertRaises(ValueError):
            P.panel_label(26)


class TestInPanelLabels(unittest.TestCase):
    """The in-axes a/b/c convention used by every multi-panel canonical figure."""

    def test_letter_sequence_is_bare(self):
        self.assertEqual([P.panel_letter(i) for i in range(3)], ["a", "b", "c"])

    def test_letter_out_of_range(self):
        with self.assertRaises(ValueError):
            P.panel_letter(-1)
        with self.assertRaises(ValueError):
            P.panel_letter(26)

    def test_profiles_are_inside_the_axes(self):
        for name, placement in P.PANEL_LABEL_PROFILES.items():
            with self.subTest(profile=name):
                self.assertTrue(0.0 < placement["x"] < 1.0, name)
                self.assertTrue(0.0 < placement["y"] < 1.0, name)
                self.assertIn(placement["va"], {"top", "bottom"})

    def test_unknown_profile_rejected(self):
        with self.assertRaises(ValueError):
            P.panel_label_placement("no_such_profile")

    def test_placement_copy_is_not_shared_state(self):
        first = P.panel_label_placement("field_triptych")
        first["x"] = 0.99
        self.assertNotEqual(P.panel_label_placement("field_triptych")["x"], 0.99)

    def test_add_panel_labels_letters_in_order(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1)
        try:
            artists = P.add_panel_labels(axes)
            self.assertEqual([a.get_text() for a in artists], ["a", "b", "c"])
            # identical placement and typography across the panels
            self.assertEqual({a.get_position() for a in artists}, {(0.02, 0.95)})
            self.assertEqual({a.get_fontweight() for a in artists}, {"normal"})
            self.assertEqual({a.get_va() for a in artists}, {"top"})
        finally:
            plt.close(fig)

    def test_profile_and_explicit_override(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        try:
            low = P.add_panel_label(ax, "a", profile="descending_line_panel")
            self.assertEqual(low.get_va(), "bottom")
            explicit = P.add_panel_label(ax, "b", x=0.5, y=0.5)
            self.assertEqual(explicit.get_position(), (0.5, 0.5))
        finally:
            plt.close(fig)


class TestModelNaming(unittest.TestCase):
    def test_display_names_and_suffixes(self):
        for model in ("NGRC", "RC", "PERC", "PINN"):
            self.assertEqual(P.model_display_name(model, "full64"), f"{model}-64")
            self.assertEqual(P.model_display_name(model, "latent8"), f"{model}-8")

    def test_prediction_symbol_representation_aware(self):
        self.assertEqual(P.prediction_symbol("NGRC", "full64"), r"\hat{u}_{\mathrm{NGRC-64}}")
        self.assertEqual(P.prediction_symbol("NGRC", "latent8"), r"\hat{u}_{\mathrm{NGRC-8}}")
        # a latent forecast must never carry -64
        for model in ("NGRC", "RC", "PERC", "PINN"):
            self.assertNotIn("-64", P.prediction_symbol(model, "latent8"))

    def test_invalid_representation(self):
        with self.assertRaises(ValueError):
            P.model_display_name("NGRC", "latent6")


class TestOptimizerDisplayMappings(unittest.TestCase):
    def setUp(self):
        from forecasting.visualization import search_plots as SP

        self.SP = SP

    def test_param_labels_are_lowercase_sentence_case(self):
        """Axis labels are lowercase; the override table only carries maths now.

        The table previously re-capitalised five of the six parameter labels the
        plot configs already store in lowercase, which is why "quadratic mode"
        -- absent from it -- rendered differently from its neighbours on the same
        matrix. Under the style contract the configs' own lowercase wording is
        used directly and only the ridge-alpha label needs a LaTeX override.
        """
        overrides = self.SP._PARAM_LABEL_OVERRIDES
        self.assertEqual(set(overrides), {"log10 ridge alpha"})
        self.assertIn(r"\alpha_{\mathrm{ridge}}", overrides["log10 ridge alpha"])

        from forecasting.visualization.search_plots import _display_param_label

        class _Spec:
            def __init__(self, label):
                self.label = label

        for label in ("feature set", "include bias", "context steps",
                      "delay spacing", "quadratic mode"):
            self.assertEqual(_display_param_label(_Spec(label)), label)

    def test_categorical_value_display(self):
        d = self.SP._display_category
        # Feature-set names render wrapped on both axes (see
        # TestFeatureSetTickLabels); the canonical single-line wording is still
        # the semantic mapping the wrapped forms are derived from.
        self.assertEqual(d("linear_quadratic"), "Linear +\nquadratic")
        self.assertEqual(d("full"), "Full\nlibrary")
        self.assertEqual(self.SP._CATEGORICAL_VALUE_DISPLAY["linear_quadratic"], "Linear + quadratic")
        self.assertEqual(self.SP._CATEGORICAL_VALUE_DISPLAY["full"], "Full library")
        self.assertEqual(d("true"), "Yes")
        self.assertEqual(d("false"), "No")
        # unknown values pass through unchanged
        self.assertEqual(d("linear"), "linear")


class TestFeatureSetTickLabels(unittest.TestCase):
    """Axis-aware categorical tick display (optimiser sampling matrix)."""

    def setUp(self):
        from forecasting.visualization import search_plots as SP

        self.SP = SP

    def test_internal_stored_values_unchanged(self):
        # The keys are the raw stored values; they must remain exactly these.
        for stored in ("full", "linear_quadratic", "true", "false"):
            self.assertIn(stored, self.SP._CATEGORICAL_VALUE_DISPLAY)
        self.assertEqual(
            set(self.SP._CATEGORICAL_VALUE_DISPLAY_WRAPPED), {"full", "linear_quadratic"}
        )

    def test_x_axis_display_is_two_line(self):
        d = self.SP._display_category
        self.assertEqual(d("full", axis="x"), "Full\nlibrary")
        self.assertEqual(d("linear_quadratic", axis="x"), "Linear +\nquadratic")

    def test_y_axis_display_is_two_line(self):
        d = self.SP._display_category
        self.assertEqual(d("full", axis="y"), "Full\nlibrary")
        self.assertEqual(d("linear_quadratic", axis="y"), "Linear +\nquadratic")

    def test_both_axes_use_identical_feature_set_text(self):
        d = self.SP._display_category
        for value in ("full", "linear_quadratic"):
            self.assertEqual(d(value, axis="x"), d(value, axis="y"))
            self.assertEqual(d(value, axis="y").count("\n"), 1)

    def test_default_axis_matches_wrapped_form(self):
        self.assertEqual(self.SP._display_category("full"), "Full\nlibrary")

    def test_boolean_display_unchanged_on_both_axes(self):
        d = self.SP._display_category
        for axis in ("x", "y"):
            self.assertEqual(d("true", axis=axis), "Yes")
            self.assertEqual(d("false", axis=axis), "No")
            self.assertNotIn("\n", d("true", axis=axis))
            self.assertNotIn("\n", d("false", axis=axis))

    def test_invalid_axis_rejected(self):
        with self.assertRaises(ValueError):
            self.SP._display_category("full", axis="z")

    def test_multiline_detection_on_both_axes(self):
        f = self.SP._is_multiline_category
        for axis in ("x", "y"):
            self.assertTrue(f({0: "full", 1: "linear_quadratic"}, axis=axis))
            self.assertFalse(f({0: "false", 1: "true"}, axis=axis))
            self.assertFalse(f(None, axis=axis))
            self.assertFalse(f({}, axis=axis))

    def test_categorical_x_ticks_are_upright_and_centred(self):
        """The bottom-row categorical x-ticks use rotation 0 / ha=center."""
        src = _read(_REPO_ROOT / "forecasting" / "src" / "forecasting" / "visualization" / "search_plots.py")
        tree = ast.parse(src)

        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "set_xticklabels"
        ]
        formatted = [c for c in calls if c.keywords]
        self.assertTrue(formatted, "expected a formatted set_xticklabels call")

        for call in formatted:
            kwargs = {kw.arg: kw.value for kw in call.keywords}
            self.assertIn("rotation", kwargs)
            self.assertEqual(ast.literal_eval(kwargs["rotation"]), 0)
            self.assertEqual(ast.literal_eval(kwargs["ha"]), "center")
            # the old anchored-rotation styling must be gone
            self.assertNotIn("rotation_mode", kwargs)

    def test_no_rotated_categorical_x_ticks_remain(self):
        src = _read(_REPO_ROOT / "forecasting" / "src" / "forecasting" / "visualization" / "search_plots.py")
        self.assertNotIn('rotation=30', src)
        self.assertNotIn('rotation_mode="anchor"', src)

    def test_y_ticks_remain_horizontal_and_right_aligned(self):
        src = _read(_REPO_ROOT / "forecasting" / "src" / "forecasting" / "visualization" / "search_plots.py")
        tree = ast.parse(src)

        formatted = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "set_yticklabels"
            and node.keywords
        ]
        self.assertTrue(formatted, "expected a formatted set_yticklabels call")

        for call in formatted:
            kwargs = {kw.arg: kw.value for kw in call.keywords}
            self.assertEqual(ast.literal_eval(kwargs["rotation"]), 0)
            self.assertEqual(ast.literal_eval(kwargs["ha"]), "right")
            self.assertEqual(ast.literal_eval(kwargs["va"]), "center")
            # wrapped lines centred with respect to each other
            self.assertEqual(ast.literal_eval(kwargs["multialignment"]), "center")

    def test_y_tick_padding_is_positive_and_applied(self):
        src = _read(_REPO_ROOT / "forecasting" / "src" / "forecasting" / "visualization" / "search_plots.py")
        self.assertGreater(self.SP._CATEGORICAL_TICK_PAD, 0.0)
        self.assertIn('ax.tick_params(axis="y", pad=_CATEGORICAL_TICK_PAD)', src)

    def test_left_margin_sufficient_and_bounded(self):
        # Wrapped y labels are narrower than the single-line form, so the left
        # margin is reduced -- but it must stay wide enough to hold them and
        # must not collapse to zero.
        left = self.SP._MATRIX_LEFT
        self.assertGreater(left, 0.05)
        self.assertLess(left, 0.20)

    def test_star_margins_and_scientific_arrays_untouched(self):
        src = _read(_REPO_ROOT / "forecasting" / "src" / "forecasting" / "visualization" / "search_plots.py")
        # star-visibility data margins preserved from the earlier approved pass
        self.assertIn("ax.margins(x=0.08, y=0.08)", src)
        # the selected point is still read from the recorded best row
        self.assertIn("_best_row(rows, metric, objective)", src)
        self.assertIn('marker="*"', src)

    def test_axis_label_spacing_is_deliberate_and_nonzero(self):
        SP = self.SP
        self.assertGreater(SP._CATEGORICAL_TICK_PAD, 0.0)
        self.assertGreater(SP._XLABEL_PAD_SINGLE_LINE, 0.0)
        self.assertGreater(SP._XLABEL_PAD_MULTILINE, 0.0)
        # two-line ticks must be given strictly more room than single-line ones
        self.assertGreater(SP._XLABEL_PAD_MULTILINE, SP._XLABEL_PAD_SINGLE_LINE)

    def test_colorbar_anchored_to_matrix_band(self):
        # Colorbar stays aligned with the panel matrix rather than a stale
        # hardcoded bottom that no longer matches the margin.
        SP = self.SP
        self.assertGreater(SP._MATRIX_TOP, SP._MATRIX_BOTTOM)
        src = _read(_REPO_ROOT / "forecasting" / "src" / "forecasting" / "visualization" / "search_plots.py")
        self.assertIn("_MATRIX_TOP - _MATRIX_BOTTOM", src)

    def test_no_raw_categorical_text_survives_display(self):
        d = self.SP._display_category
        for axis in ("x", "y"):
            for stored in ("full", "linear_quadratic", "true", "false"):
                rendered = d(stored, axis=axis)
                self.assertNotIn("linear_quadratic", rendered)
                self.assertNotIn("_", rendered)
                self.assertNotEqual(rendered, "true")
                self.assertNotEqual(rendered, "false")

    def test_display_helper_does_not_mutate_inputs(self):
        stored_rows = [{"feature_set": "full"}, {"feature_set": "linear_quadratic"}]
        snapshot = [dict(row) for row in stored_rows]
        for row in stored_rows:
            self.SP._display_category(row["feature_set"], axis="x")
            self.SP._display_category(row["feature_set"], axis="y")
        self.assertEqual(stored_rows, snapshot)

    def test_stored_values_survive_parameter_extraction(self):
        """The categorical encoder still returns the raw stored labels."""
        SP = self.SP
        spec = SP.ParameterSpec(name="feature_set", label="feature set", kind="categorical")
        rows = [{"feature_set": "full"}, {"feature_set": "linear_quadratic"}]
        values, ticks = SP._parameter_values(rows, spec)
        self.assertEqual(sorted(ticks.values()), ["full", "linear_quadratic"])
        self.assertEqual(len(values), 2)


class TestSurrogateColorbarFormat(unittest.TestCase):
    """Colorbar tick formatting for the optimiser surrogate landscapes.

    Policy under test: never show Matplotlib's detached additive offset. Use
    fixed-point labels with just enough decimals to separate adjacent ticks;
    when that would need more than ``_COLORBAR_MAX_DECIMALS`` — i.e. the mapped
    surface is constant to ordinary precision, as in ``pinn_full64`` — show three
    complete absolute-value ticks at vmin, midpoint and vmax at the least
    precision that keeps them distinct, never a single centred tick.
    """

    def setUp(self):
        import matplotlib

        matplotlib.use("Agg")
        from forecasting.visualization import search_plots as SP

        self.SP = SP

    def _colorbar(self, vmin, vmax):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        data = np.linspace(vmin, vmax, 64).reshape(8, 8)
        image = ax.imshow(data)
        image.set_clim(vmin, vmax)
        cax = fig.add_axes([0.915, 0.10, 0.016, 0.82])
        return fig, image, fig.colorbar(mappable=image, cax=cax, label="Test")

    def _labels(self, colorbar):
        colorbar.ax.figure.canvas.draw()
        return [t.get_text() for t in colorbar.ax.get_yticklabels()]

    def _offset(self, colorbar):
        colorbar.ax.figure.canvas.draw()
        return colorbar.ax.yaxis.get_offset_text().get_text()

    # --- degenerate (pinn_full64) range -------------------------------------

    # A pinn_full64-shaped clim: constant to ~9 decimals, varying at the 10th.
    # The default ScalarFormatter renders this as "1e-10 + 1.496544027e1".
    PINN_VMIN = 14.965440270282107
    PINN_VMAX = 14.965440271065933

    def _ticks(self, colorbar):
        colorbar.ax.figure.canvas.draw()
        return [float(t) for t in colorbar.get_ticks()]

    def test_degenerate_range_leaves_no_offset_text(self):
        import matplotlib.pyplot as plt

        fig, _, cb = self._colorbar(self.PINN_VMIN, self.PINN_VMAX)
        self.assertNotEqual(self._offset(cb), "")  # default behaviour, pre-fix
        self.SP._apply_colorbar_tick_format(cb)
        self.assertEqual(self._offset(cb), "")
        plt.close(fig)

    def test_degenerate_range_has_exactly_three_ticks(self):
        import matplotlib.pyplot as plt

        fig, _, cb = self._colorbar(self.PINN_VMIN, self.PINN_VMAX)
        decision = self.SP._apply_colorbar_tick_format(cb)
        self.assertTrue(decision["degenerate"])
        labels = [t for t in self._labels(cb) if t]
        self.assertEqual(len(labels), 3, labels)
        plt.close(fig)

    def test_degenerate_ticks_are_vmin_mid_vmax(self):
        import matplotlib.pyplot as plt

        fig, _, cb = self._colorbar(self.PINN_VMIN, self.PINN_VMAX)
        self.SP._apply_colorbar_tick_format(cb)
        ticks = self._ticks(cb)
        midpoint = 0.5 * (self.PINN_VMIN + self.PINN_VMAX)
        self.assertEqual(len(ticks), 3)
        self.assertAlmostEqual(ticks[0], self.PINN_VMIN, places=12)
        self.assertAlmostEqual(ticks[1], midpoint, places=12)
        self.assertAlmostEqual(ticks[2], self.PINN_VMAX, places=12)
        plt.close(fig)

    def test_degenerate_labels_are_complete_absolute_values(self):
        import matplotlib.pyplot as plt

        fig, _, cb = self._colorbar(self.PINN_VMIN, self.PINN_VMAX)
        self.SP._apply_colorbar_tick_format(cb)
        labels = [t for t in self._labels(cb) if t]
        # Absolute values near 14.96, not residuals near 0 and not an offset.
        for lab in labels:
            self.assertIn(".", lab)
            self.assertGreater(float(lab), 14.0)
            self.assertLess(float(lab), 16.0)
        plt.close(fig)

    def test_degenerate_labels_are_distinct(self):
        import matplotlib.pyplot as plt

        fig, _, cb = self._colorbar(self.PINN_VMIN, self.PINN_VMAX)
        self.SP._apply_colorbar_tick_format(cb)
        labels = [t for t in self._labels(cb) if t]
        self.assertEqual(len(labels), len(set(labels)), f"identical labels: {labels}")
        plt.close(fig)

    def test_degenerate_labels_reconstruct_tick_values(self):
        import matplotlib.pyplot as plt

        fig, _, cb = self._colorbar(self.PINN_VMIN, self.PINN_VMAX)
        self.SP._apply_colorbar_tick_format(cb)
        ticks = self._ticks(cb)
        labels = [t for t in self._labels(cb) if t]
        decimals = len(labels[0].split(".")[1])
        tolerance = 0.5 * 10 ** (-decimals)
        for tick, lab in zip(ticks, labels):
            # Every label parses back to its tick within half a display unit.
            self.assertLessEqual(abs(float(lab) - tick), tolerance, (lab, tick))
        plt.close(fig)

    def test_no_single_tick_degenerate_branch_remains(self):
        src = _read(_REPO_ROOT / "forecasting/src/forecasting/visualization/search_plots.py")
        body = src.split("def _apply_colorbar_tick_format", 1)[1].split("\ndef ", 1)[0]
        # A three-tick vmin/midpoint/vmax list must be constructed.
        self.assertIn("[vmin, midpoint, vmax]", body)

    def test_formatter_does_not_change_clim(self):
        import matplotlib.pyplot as plt

        fig, image, cb = self._colorbar(self.PINN_VMIN, self.PINN_VMAX)
        before = image.get_clim()
        self.SP._apply_colorbar_tick_format(cb)
        self.assertEqual(image.get_clim(), before)
        plt.close(fig)

    def test_formatter_does_not_change_norm_or_array(self):
        import matplotlib.pyplot as plt

        fig, image, cb = self._colorbar(self.PINN_VMIN, self.PINN_VMAX)
        norm_before = (image.norm.vmin, image.norm.vmax, type(image.norm))
        array_before = np.array(image.get_array(), copy=True)
        self.SP._apply_colorbar_tick_format(cb)
        self.assertEqual((image.norm.vmin, image.norm.vmax, type(image.norm)), norm_before)
        np.testing.assert_array_equal(np.asarray(image.get_array()), array_before)
        plt.close(fig)

    # --- ordinary ranges stay concise ---------------------------------------

    def test_ordinary_range_labels_stay_integer_like(self):
        import matplotlib.pyplot as plt

        fig, _, cb = self._colorbar(170.6354827, 302.9441258)
        self.SP._apply_colorbar_tick_format(cb)
        labels = [t for t in self._labels(cb) if t]
        self.assertTrue(all("." not in t for t in labels), labels)
        self.assertEqual(self._offset(cb), "")
        plt.close(fig)

    def test_ordinary_range_is_not_treated_as_degenerate(self):
        import matplotlib.pyplot as plt

        fig, _, cb = self._colorbar(6.20965332, 20.68279827)
        decision = self.SP._apply_colorbar_tick_format(cb)
        self.assertFalse(decision["degenerate"])
        self.assertEqual(decision["decimals"], 0)
        plt.close(fig)

    def test_fine_range_gains_decimals_without_offset(self):
        import matplotlib.pyplot as plt

        fig, _, cb = self._colorbar(14.960, 14.985)
        decision = self.SP._apply_colorbar_tick_format(cb)
        self.assertFalse(decision["degenerate"])
        self.assertGreater(decision["decimals"], 0)
        labels = [t for t in self._labels(cb) if t]
        self.assertEqual(len(labels), len(set(labels)), labels)
        self.assertEqual(self._offset(cb), "")
        plt.close(fig)

    # --- helper policy ------------------------------------------------------

    def test_tick_decimals_policy(self):
        d = self.SP._tick_decimals
        self.assertEqual(d(20.0), 0)
        self.assertEqual(d(2.0), 0)
        self.assertEqual(d(1.0), 0)
        self.assertEqual(d(0.5), 1)
        self.assertEqual(d(0.02), 2)
        self.assertEqual(d(1e-10), 10)
        self.assertEqual(d(0.0), 0)

    def test_concise_value_trims_and_avoids_negative_zero(self):
        c = self.SP._concise_value
        self.assertEqual(c(160.0, 0), "160")
        self.assertEqual(c(14.965440269, 6), "14.96544")
        self.assertEqual(c(-0.0, 2), "0")
        self.assertEqual(c(-0.004, 2), "0")

    def test_precision_cap_is_bounded(self):
        self.assertLessEqual(self.SP._COLORBAR_MAX_DECIMALS, 6)
        self.assertGreater(self.SP._COLORBAR_DEGENERATE_REL_SPAN, 0.0)
        # The degenerate search must stay within float64's honest precision.
        self.assertGreaterEqual(self.SP._COLORBAR_DEGENERATE_MAX_DECIMALS, 10)
        self.assertLessEqual(self.SP._COLORBAR_DEGENERATE_MAX_DECIMALS, 15)

    def test_distinct_fixed_decimals_policy(self):
        f = self.SP._distinct_fixed_decimals
        # pinn_full64-shaped triple needs 10 decimals to separate.
        vmin, vmax = 14.965440270282107, 14.965440271065933
        mid = 0.5 * (vmin + vmax)
        self.assertEqual(f([vmin, mid, vmax], max_decimals=12), 10)
        # Already-distinct integers need none.
        self.assertEqual(f([1.0, 2.0, 3.0], max_decimals=12), 0)
        # Truly equal values cannot be separated.
        self.assertIsNone(f([5.0, 5.0, 5.0], max_decimals=6))

    def test_surrogate_plot_applies_the_formatter(self):
        src = _read(_REPO_ROOT / "forecasting/src/forecasting/visualization/search_plots.py")
        body = src.split("def plot_surrogate_landscape", 1)[1]
        self.assertIn("_apply_colorbar_tick_format(colorbar)", body)

    def test_colorbar_geometry_constants_unchanged(self):
        """The approved layout: absolute colorbar axes, unchanged margins."""
        src = _read(_REPO_ROOT / "forecasting/src/forecasting/visualization/search_plots.py")
        # The colourbar axes is still placed absolutely and the margins are still
        # fixed; the two x constants moved left in Phase 3 because they are
        # FRACTIONS of the canvas. At the historical 12 in width the 0.069 left
        # of the colourbar was 0.83 in and the rotated label fitted; at the
        # printed 5.9 in it is 0.41 in and the label overflowed the figure,
        # which bbox_inches="tight" then absorbed by growing the saved canvas.
        self.assertIn("fig.add_axes([0.875, 0.10, 0.016, 0.82])", src)
        # Phase 3 moved three of these four margins, each for a measured reason
        # at the printed canvas size:
        #   right 0.90  -> 0.855   the rotated colourbar label overflowed
        #   wspace 0.16 -> 0.24    outer numeric ticks touched the next panel
        #                          (measured sweep: 0.16 -> 3 collisions, 0.24 -> 0)
        #   left 0.11   -> 0.145   a two-line rotated y-label plus its ticks
        #                          needed more than the 0.65 in that 0.11 gave
        # The layout is still absolutely placed with fixed margins, which is what
        # this test exists to protect.
        self.assertIn(
            "fig.subplots_adjust(left=0.145, right=0.855, bottom=0.10, top=0.92, "
            "wspace=0.24, hspace=0.18)",
            src,
        )
        # CH-13: the canvas is now the PRINTED canvas, so the literal inch
        # formula moved into _surrogate_figsize, which keeps the historical
        # proportions while sizing to the page. A figure drawn larger than it is
        # printed has its type sizes reduced by LaTeX, which is the defect this
        # change exists to remove.
        self.assertIn("figsize=_surrogate_figsize(n)", src)
        self.assertIn("(2.65 * n + 1.15) / (2.85 * n + 1.35)", src)

        from common import plot_style as PS

        for n in (3, 4, 6):
            width, height = self.SP._surrogate_figsize(n) if hasattr(self, "SP") \
                else __import__(
                    "forecasting.visualization.search_plots",
                    fromlist=["_surrogate_figsize"])._surrogate_figsize(n)
            self.assertAlmostEqual(width, PS.page_class("full_width")["width_in"], 6)
            expected = width * (2.65 * n + 1.15) / (2.85 * n + 1.35)
            self.assertAlmostEqual(height, expected, 6)

    def test_formatter_uses_no_case_specific_logic(self):
        """The policy must be range-driven, never keyed on a case or file name."""
        src = _read(_REPO_ROOT / "forecasting/src/forecasting/visualization/search_plots.py")
        body = src.split("def _apply_colorbar_tick_format", 1)[1].split("\ndef ", 1)[0]
        for token in ("pinn", "full64", "latent8", "ngrc", "perc"):
            self.assertNotIn(token, body.lower())

    def test_formatter_reads_no_private_path(self):
        src = _read(_REPO_ROOT / "forecasting/src/forecasting/visualization/search_plots.py")
        for token in ("/home/", "/scratch/", "DataDrivenModelsOfChaoticPDEs", "dsnf120h"):
            self.assertNotIn(token, src)


class TestSeedRobustnessPort(unittest.TestCase):
    def setUp(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "plot_seed_robustness", _REPO_ROOT / "forecasting" / "scripts" / "plot_seed_robustness.py"
        )
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_six_supported_models(self):
        # pinn_full64 joined the set when its v003 physics-active study replaced the earlier
        # staged rerank with a uniform 10 x 15 grid. NGRC is deterministic (no RNG), so it
        # still has no seed distribution and must stay out.
        self.assertEqual(len(self.mod.SEED_MODELS), 6)
        self.assertIn("pinn_full64", self.mod.SEED_MODELS)
        self.assertNotIn("ngrc_full64", self.mod.SEED_MODELS)
        self.assertNotIn("ngrc_latent8", self.mod.SEED_MODELS)

    def test_legend_entries(self):
        self.assertEqual(self.mod.LEGEND_ENTRIES, ("Winner", "Published candidate", "Other candidates"))

    def test_nice_model_name(self):
        self.assertEqual(self.mod.nice_model_name("rc_full64"), "RC-64")
        self.assertEqual(self.mod.nice_model_name("pinn_latent8"), "PINN-8")

    def test_deterministic_candidate_order(self):
        cand = {r: [float(r)] for r in range(1, 11)}
        ordered = self.mod.ordered_candidates(cand)
        self.assertEqual(ordered, [[float(r)] for r in range(1, 11)])

    def test_no_private_absolute_path(self):
        src = (_REPO_ROOT / "forecasting" / "scripts" / "plot_seed_robustness.py").read_text()
        self.assertNotIn("DataDrivenModelsOfChaoticPDEs", src)
        self.assertNotIn("/home/", src)


class TestSourceHygiene(unittest.TestCase):
    PLOT_LIBRARY_MODULES = (
        "common/plotting.py",
        "forecasting/src/forecasting/visualization/forecast_plots.py",
        "forecasting/src/forecasting/visualization/search_plots.py",
        "forecasting/src/forecasting/visualization/style.py",
        "ae_sindy/ae_sindy/visualization.py",
    )

    def test_no_legacy_composite_labels(self):
        # No (a.1)/(e.1)/(f) composite scheme survives in the AE plotting source.
        src = (_REPO_ROOT / "ae_sindy" / "ae_sindy" / "visualization.py").read_text()
        self.assertNotIn("{title_prefix}.", src)
        self.assertFalse(re.search(r"\(\{title_prefix\}\)", src))

    def test_no_hardcoded_minus64_in_forecast(self):
        src = (_REPO_ROOT / "forecasting" / "src" / "forecasting" / "visualization" / "forecast_plots.py").read_text()
        # No hardcoded full-state suffix in a label format string; the symbol is
        # produced by the representation-aware shared helper instead.
        self.assertNotIn("model_name}-64", src)
        self.assertNotIn(r"\text{", src)
        self.assertIn("prediction_symbol", src)

    def test_no_descriptive_kse_title(self):
        src = (_REPO_ROOT / "simulation" / "kse_simulation.py").read_text()
        self.assertNotIn("KSE Spatiotemporal Evolution", src)
        self.assertNotIn("Time [Lyapunov units]", src)

    def test_no_seed_figure_title(self):
        # the public seed port must not call set_title / suptitle
        src = (_REPO_ROOT / "forecasting" / "scripts" / "plot_seed_robustness.py").read_text()
        tree = ast.parse(src)
        calls = [
            n.func.attr
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        ]
        self.assertNotIn("set_title", calls)
        self.assertNotIn("suptitle", calls)

    def test_save_padding_nonzero_default(self):
        from forecasting.visualization.style import save_figure
        import inspect

        default = inspect.signature(save_figure).parameters["pad_inches"].default
        self.assertGreater(default, 0.0)

    def test_no_syspath_mutation_in_plot_libraries(self):
        for rel in self.PLOT_LIBRARY_MODULES:
            src = (_REPO_ROOT / rel).read_text()
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in {"insert", "append"}:
                    value = node.value
                    if isinstance(value, ast.Attribute) and value.attr == "path":
                        self.fail(f"sys.path mutation found in plotting library module {rel}")


class TestTriptychColorScales(unittest.TestCase):
    """The repository-wide true/prediction/error colour-scale convention.

    Panels (a) and (b) share ONE symmetric range derived from the displayed true field
    alone; panel (c) gets an independent symmetric error range. These tests pin the
    behaviour that makes a diverging prediction saturate instead of flattening the truth.
    """

    def test_truth_limits_are_symmetric_about_zero(self):
        vmin, vmax = P.truth_prediction_limits(np.array([[-2.0, 1.0], [3.0, 0.5]]))
        self.assertEqual(vmin, -vmax)
        self.assertGreater(vmax, 0.0)

    def test_truth_limits_equal_nanmax_abs_of_truth(self):
        truth = np.array([[-4.25, 1.0], [3.0, 0.5]])
        _, vmax = P.truth_prediction_limits(truth)
        self.assertEqual(vmax, 4.25)

    def test_truth_limits_ignore_nan(self):
        truth = np.array([[np.nan, 1.0], [3.0, np.nan]])
        _, vmax = P.truth_prediction_limits(truth)
        self.assertEqual(vmax, 3.0)

    def test_prediction_outliers_do_not_widen_the_shared_range(self):
        """The whole point: a diverging prediction must not rescale the truth panel."""
        truth = np.array([[-2.0, 1.0], [3.0, 0.5]])
        baseline = P.truth_prediction_limits(truth)
        for scale in (1e2, 1e6, 1e12):
            wild = truth * scale
            self.assertEqual(
                P.truth_prediction_limits(truth), baseline,
                msg=f"shared range changed when the prediction was scaled by {scale}",
            )
            # the helper takes only the truth, so a prediction cannot even be passed in
            self.assertEqual(P.truth_prediction_limits(truth)[1], baseline[1])
            del wild

    def test_error_limits_are_independent_and_symmetric(self):
        truth = np.array([[1.0, 1.0], [1.0, 1.0]])
        pred = np.array([[1.0, 1.0], [1.0, -99.0]])
        fvmin, fvmax = P.truth_prediction_limits(truth)
        evmin, evmax = P.error_limits(truth, pred)
        self.assertEqual(evmin, -evmax)
        self.assertEqual(evmax, 100.0)
        self.assertNotEqual(evmax, fvmax)

    def test_error_limits_equal_nanmax_abs_of_error(self):
        truth = np.array([[0.0, 2.0]])
        pred = np.array([[-3.0, 2.0]])
        _, evmax = P.error_limits(truth, pred)
        self.assertEqual(evmax, 3.0)

    def test_identically_zero_field_uses_the_nonzero_fallback(self):
        vmin, vmax = P.truth_prediction_limits(np.zeros((4, 4)))
        self.assertEqual(vmax, P.MIN_COLOR_HALF_RANGE)
        self.assertEqual(vmin, -P.MIN_COLOR_HALF_RANGE)
        self.assertLess(vmin, vmax)  # never a singular normalisation

    def test_zero_error_uses_the_nonzero_fallback(self):
        field = np.ones((3, 3))
        vmin, vmax = P.error_limits(field, field)
        self.assertEqual(vmax, P.MIN_COLOR_HALF_RANGE)
        self.assertLess(vmin, vmax)

    def test_non_finite_truth_fails_loudly(self):
        with self.assertRaises(ValueError):
            P.truth_prediction_limits(np.full((3, 3), np.nan))
        with self.assertRaises(ValueError):
            P.truth_prediction_limits(np.array([[np.inf, -np.inf]]))

    def test_non_finite_error_fails_loudly(self):
        with self.assertRaises(ValueError):
            P.error_limits(np.full((2, 2), np.nan), np.zeros((2, 2)))

    def test_helpers_do_not_mutate_their_inputs(self):
        truth = np.array([[1.0, -2.0], [3.0, 4.0]])
        pred = np.array([[0.0, 0.0], [0.0, 0.0]])
        t0, p0 = truth.copy(), pred.copy()
        P.truth_prediction_limits(truth)
        P.error_limits(truth, pred)
        P.prediction_saturation(truth, pred)
        np.testing.assert_array_equal(truth, t0)
        np.testing.assert_array_equal(pred, p0)

    def test_saturation_is_reported_not_corrected(self):
        truth = np.array([[1.0, -1.0]])
        pred = np.array([[5.0, 0.0]])
        sat = P.prediction_saturation(truth, pred)
        self.assertEqual(sat["field_limit"], 1.0)
        self.assertEqual(sat["n_saturated"], 1)
        self.assertEqual(sat["saturation_fraction"], 0.5)
        self.assertEqual(sat["prediction_max"], 5.0)
        # the field limit is unchanged by the saturating prediction
        self.assertEqual(P.truth_prediction_limits(truth)[1], 1.0)

    def test_saturation_ignores_non_finite_predictions(self):
        truth = np.array([[1.0, 1.0]])
        pred = np.array([[np.nan, 5.0]])
        sat = P.prediction_saturation(truth, pred)
        self.assertEqual(sat["n_finite_prediction"], 1)
        self.assertEqual(sat["n_nonfinite_prediction"], 1)
        self.assertEqual(sat["saturation_fraction"], 1.0)


class TestTriptychSourcesUseSharedHelper(unittest.TestCase):
    """Every triptych generator must use the shared helper, with no local rule."""

    TRIPTYCH_SOURCES = (
        "forecasting/src/forecasting/visualization/forecast_plots.py",
        "ae_sindy/ae_sindy/visualization.py",
    )

    def _sources(self):
        for rel in self.TRIPTYCH_SOURCES:
            yield rel, _read(_REPO_ROOT / rel)

    def test_every_triptych_source_calls_the_shared_helper(self):
        for rel, body in self._sources():
            self.assertIn("truth_prediction_limits", body,
                          f"{rel} does not derive field limits from the shared helper")
            self.assertIn("error_limits", body,
                          f"{rel} does not derive error limits from the shared helper")

    def test_no_percentile_based_field_or_error_scaling(self):
        for rel, body in self._sources():
            self.assertNotIn("nanpercentile", body,
                             f"{rel} still uses a percentile-based colour limit")

    def test_no_combined_truth_prediction_extrema(self):
        """A field range must never be built from the truth+prediction union."""
        for rel, body in self._sources():
            self.assertNotIn("min(np.nanmin(x_true), np.nanmin(x_pred))", body, rel)
            self.assertNotIn("max(np.nanmax(x_true), np.nanmax(x_pred))", body, rel)

    def test_forecasting_panels_ab_share_one_normalisation(self):
        body = _read(_REPO_ROOT / self.TRIPTYCH_SOURCES[0])
        # the truth and prediction panel tuples must carry the same vmin/vmax names
        self.assertIn("(r\"$u(x,t)$\", truth_plot, vmin_state, vmax_state,", body)
        self.assertIn("pred_plot, vmin_state, vmax_state,", body)
        # and the error panel must use its own
        self.assertIn("vmin_error", body)
        self.assertIn("vmax_error", body)


_PLOT_CONFIG = (
    _REPO_ROOT / "final_thesis_package_v001" / "configs" / "03_plots" / "forecasting_visuals.json"
)


def _selected_runs():
    """The tracked selected runs the public forecasting figures are drawn from."""
    if not _PLOT_CONFIG.is_file():
        return []
    cfg = json.loads(_PLOT_CONFIG.read_text(encoding="utf-8"))
    return cfg.get("inputs", {}).get("selected_runs", [])


def _write_run(root: Path, split: str, *, metrics: dict, config: dict | None = None,
               n_time: int = 20, n_space: int = 8) -> Path:
    """Build a throwaway run directory in the shape the plotting layer expects."""
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    truth = rng.normal(size=(n_time, n_space))
    np.save(split_dir / "truth_physical.npy", truth)
    np.save(split_dir / "pred_physical.npy", truth + 0.25)

    (split_dir / "metrics_physical.json").write_text(json.dumps(metrics), encoding="utf-8")
    if config is not None:
        (root / "config_resolved.json").write_text(json.dumps(config), encoding="utf-8")

    return root


class TestForecastDtResolution(unittest.TestCase):
    """The display sampling interval must be resolved, never assumed.

    A wrong dt rescales the whole time axis while every tick label still looks
    plausible, so the resolver has an explicit precedence and no numeric
    default: explicit argument, then metrics, then the resolved run config that
    produced those metrics, then a raise.
    """

    def setUp(self):
        from forecasting.visualization import forecast_plots as FP

        self.FP = FP

    def test_explicit_dt_has_highest_precedence(self):
        resolved = self.FP._resolve_dt(
            run_dir=Path("/nonexistent"), metrics={"dt": 0.1}, explicit_dt=0.25
        )
        self.assertEqual(resolved, 0.25)

    def test_metrics_dt_used_when_no_explicit_value(self):
        resolved = self.FP._resolve_dt(run_dir=Path("/nonexistent"), metrics={"dt": 0.1})
        self.assertEqual(resolved, 0.1)

    def test_config_dt_used_when_metrics_omit_dt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config_resolved.json").write_text(
                json.dumps({"evaluation": {"dt": 0.1}}), encoding="utf-8"
            )
            self.assertEqual(self.FP._resolve_dt(run_dir=root, metrics={}), 0.1)

    def test_missing_everywhere_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self.FP._resolve_dt(run_dir=Path(tmp), metrics={})

    def test_missing_everywhere_never_returns_a_silent_default(self):
        # The two values a silent fallback would most plausibly invent.
        with tempfile.TemporaryDirectory() as tmp:
            try:
                resolved = self.FP._resolve_dt(run_dir=Path(tmp), metrics={})
            except ValueError:
                return
            self.fail(f"resolver returned {resolved!r} instead of raising")

    def test_config_without_evaluation_dt_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config_resolved.json").write_text(
                json.dumps({"evaluation": {"relative_l2_threshold": 0.5}}), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                self.FP._resolve_dt(run_dir=root, metrics={})

    def test_zero_dt_raises(self):
        with self.assertRaises(ValueError):
            self.FP._resolve_dt(run_dir=Path("/nonexistent"), metrics={"dt": 0.0})

    def test_negative_dt_raises(self):
        with self.assertRaises(ValueError):
            self.FP._resolve_dt(run_dir=Path("/nonexistent"), metrics={"dt": -0.1})

    def test_non_finite_dt_raises(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(dt=bad):
                with self.assertRaises(ValueError):
                    self.FP._resolve_dt(run_dir=Path("/nonexistent"), metrics={"dt": bad})

    def test_non_numeric_dt_raises(self):
        with self.assertRaises(ValueError):
            self.FP._resolve_dt(run_dir=Path("/nonexistent"), metrics={"dt": "fast"})

    def test_invalid_explicit_dt_raises_rather_than_falling_back(self):
        with self.assertRaises(ValueError):
            self.FP._resolve_dt(run_dir=Path("/nonexistent"), metrics={"dt": 0.1}, explicit_dt=0.0)

    def test_selected_runs_resolve_the_expected_source(self):
        runs = _selected_runs()
        if not runs:
            self.skipTest("tracked forecasting_visuals.json is unavailable")

        seen_metrics = seen_config = 0
        for run in runs:
            run_dir = _REPO_ROOT / run["run_dir"]
            if not run_dir.is_dir():
                self.skipTest(f"selected run is unavailable: {run['run_dir']}")

            for split in ("validation", "test"):
                metrics = json.loads(
                    (run_dir / split / "metrics_physical.json").read_text(encoding="utf-8")
                )
                resolved = self.FP._resolve_dt(run_dir=run_dir, metrics=metrics)
                with self.subTest(run=run["label"], split=split):
                    self.assertEqual(resolved, 0.1)
                    # RC/NGRC/PERC carry dt in metrics; PINN's leaner schema does not
                    # and must fall through to the tracked resolved run config.
                    if metrics.get("dt") is not None:
                        seen_metrics += 1
                    else:
                        self.assertEqual(run["model_name"], "PINN")
                        config = json.loads(
                            (run_dir / "config_resolved.json").read_text(encoding="utf-8")
                        )
                        self.assertEqual(config["evaluation"]["dt"], 0.1)
                        seen_config += 1

        self.assertEqual(seen_metrics, 12, "expected 12 non-PINN splits to resolve from metrics")
        self.assertEqual(seen_config, 4, "expected 4 PINN splits to resolve from the run config")

    def test_no_numeric_fallback_remains_in_the_source(self):
        body = _read(
            _REPO_ROOT / "forecasting" / "src" / "forecasting" / "visualization" / "forecast_plots.py"
        )
        self.assertNotIn('metrics.get("dt", 1.0)', body)
        self.assertNotIn('metrics.get("dt", 0.1)', body)


class TestSafetyCutoffAlias(unittest.TestCase):
    """PINN records the masking index under a different name; reading is aliased.

    ``prediction_masked_from_index`` stays authoritative. The alias exists only
    so a leaner metrics schema still draws its safety cutoff, and it never
    writes to the metrics record.
    """

    def setUp(self):
        from forecasting.visualization import forecast_plots as FP

        self.FP = FP

    def test_preferred_field_wins_when_both_present(self):
        metrics = {"prediction_masked_from_index": 588, "first_unsafe_index": 198}
        self.assertEqual(self.FP._resolve_masked_from_index(metrics), 588.0)

    def test_alias_used_when_preferred_field_absent(self):
        self.assertEqual(self.FP._resolve_masked_from_index({"first_unsafe_index": 198}), 198.0)

    def test_none_when_both_absent(self):
        self.assertIsNone(self.FP._resolve_masked_from_index({"relative_l2_horizon_time": 19.8}))

    def test_unusable_values_are_ignored(self):
        for metrics in (
            {"prediction_masked_from_index": None, "first_unsafe_index": None},
            {"first_unsafe_index": float("nan")},
            {"first_unsafe_index": -1},
            {"first_unsafe_index": "later"},
        ):
            with self.subTest(metrics=metrics):
                self.assertIsNone(self.FP._resolve_masked_from_index(metrics))

    def test_unusable_preferred_value_falls_through_to_the_alias(self):
        metrics = {"prediction_masked_from_index": None, "first_unsafe_index": 104}
        self.assertEqual(self.FP._resolve_masked_from_index(metrics), 104.0)

    def test_metrics_are_not_mutated(self):
        metrics = {"first_unsafe_index": 73}
        before = dict(metrics)
        self.FP._resolve_masked_from_index(metrics)
        self.assertEqual(metrics, before)
        self.assertNotIn("prediction_masked_from_index", metrics)


class TestHorizonLegendPlacement(unittest.TestCase):
    """One marker legend per triptych, on the error panel, lines on every panel.

    This mirrors the AE-SINDy field-triptych convention: repeating an identical
    key in all three panels crowds the field without adding information, but the
    marker lines themselves must stay everywhere they were.
    """

    HORIZON = "Prediction horizon"
    CUTOFF = "Safety cutoff"

    def setUp(self):
        import matplotlib

        matplotlib.use("Agg")
        from forecasting.visualization import forecast_plots as FP

        self.FP = FP

    def _render(self, metrics, *, config=None, split="validation"):
        """Render one triptych and hand back the figure before it is saved."""
        from unittest import mock

        captured = {}

        def _capture(fig, output_base, formats, dpi, **kwargs):
            captured["fig"] = fig
            return []

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = _write_run(Path(tmp.name), split, metrics=metrics, config=config)

        with mock.patch.object(self.FP, "save_figure", _capture):
            self.FP.plot_spatiotemporal_comparison(
                run_dir=root,
                split=split,
                model_name="RC",
                representation="full64",
                output_base=root / "out",
                formats=["png"],
                dpi=50,
                use_lyapunov=True,
                lambda_max=0.043,
            )

        fig = captured["fig"]
        self.addCleanup(self._close, fig)
        return fig

    @staticmethod
    def _close(fig):
        import matplotlib.pyplot as plt

        plt.close(fig)

    @staticmethod
    def _legend_axes(fig):
        return [i for i, ax in enumerate(fig.axes) if ax.get_legend() is not None]

    def _triptych_axes(self, fig):
        # Each panel also contributes a colorbar axes, and those carry a QuadMesh
        # too; the panels are the axes that set the shared spatial ylabel.
        return [ax for ax in fig.axes if ax.get_ylabel() == r"space $x$"]

    def test_single_legend_on_the_final_error_axis(self):
        fig = self._render({"dt": 0.1, "relative_l2_horizon_time": 1.0,
                            "prediction_masked_from_index": 5})
        panels = self._triptych_axes(fig)
        self.assertEqual(len(panels), 3)

        with_legend = [ax for ax in panels if ax.get_legend() is not None]
        self.assertEqual(len(with_legend), 1, "expected exactly one legend in the triptych")
        self.assertIs(with_legend[0], panels[-1], "the legend must sit on the error panel")

    def test_no_legend_on_truth_or_prediction_panels(self):
        fig = self._render({"dt": 0.1, "relative_l2_horizon_time": 1.0,
                            "prediction_masked_from_index": 5})
        panels = self._triptych_axes(fig)
        self.assertIsNone(panels[0].get_legend())
        self.assertIsNone(panels[1].get_legend())

    def test_horizon_line_remains_on_every_panel(self):
        fig = self._render({"dt": 0.1, "relative_l2_horizon_time": 1.0})
        for index, ax in enumerate(self._triptych_axes(fig)):
            with self.subTest(panel=index):
                dashed = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
                self.assertEqual(len(dashed), 1)

    def test_cutoff_line_and_shading_remain_on_every_panel(self):
        fig = self._render({"dt": 0.1, "relative_l2_horizon_time": 1.0,
                            "prediction_masked_from_index": 5})
        for index, ax in enumerate(self._triptych_axes(fig)):
            with self.subTest(panel=index):
                dotted = [ln for ln in ax.lines if ln.get_linestyle() == ":"]
                self.assertEqual(len(dotted), 1, "safety cutoff line missing")
                self.assertGreaterEqual(len(ax.patches), 1, "safety shading missing")

    def test_legend_carries_every_applicable_marker_entry(self):
        fig = self._render({"dt": 0.1, "relative_l2_horizon_time": 1.0,
                            "prediction_masked_from_index": 5})
        legend = self._triptych_axes(fig)[-1].get_legend()
        labels = [t.get_text() for t in legend.get_texts()]
        self.assertEqual(len(labels), 2)
        self.assertTrue(any(self.HORIZON in text for text in labels), labels)
        self.assertTrue(any(self.CUTOFF in text for text in labels), labels)

    def test_pinn_alias_produces_a_cutoff_entry_without_a_dt_in_metrics(self):
        fig = self._render(
            {"relative_l2_horizon_time": 1.0, "first_unsafe_index": 5},
            config={"evaluation": {"dt": 0.1}},
        )
        panels = self._triptych_axes(fig)
        labels = [t.get_text() for t in panels[-1].get_legend().get_texts()]
        self.assertTrue(any(self.CUTOFF in text for text in labels), labels)
        for ax in panels:
            self.assertEqual(len([ln for ln in ax.lines if ln.get_linestyle() == ":"]), 1)

    def test_marker_free_case_draws_no_empty_legend(self):
        fig = self._render({"dt": 0.1})
        for index, ax in enumerate(self._triptych_axes(fig)):
            with self.subTest(panel=index):
                self.assertIsNone(ax.get_legend())

    def test_relative_l2_horizon_keeps_its_own_single_axis_legend(self):
        from unittest import mock

        captured = {}

        def _capture(fig, output_base, formats, dpi, **kwargs):
            captured["fig"] = fig
            return []

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        split_dir = root / "validation"
        split_dir.mkdir(parents=True)
        (split_dir / "metrics_physical.json").write_text(
            json.dumps({"dt": 0.1, "relative_l2_threshold": 0.5, "relative_l2_horizon_time": 1.0}),
            encoding="utf-8",
        )
        (split_dir / "errors_physical.csv").write_text(
            "index,time,relative_l2\n" + "".join(
                f"{i},{i * 0.1},{0.05 * i}\n" for i in range(20)
            ),
            encoding="utf-8",
        )

        with mock.patch.object(self.FP, "save_figure", _capture):
            self.FP.plot_relative_l2_horizon(
                run_dir=root,
                split="validation",
                output_base=root / "out",
                formats=["png"],
                dpi=50,
                use_lyapunov=True,
                lambda_max=0.043,
            )

        fig = captured["fig"]
        self.addCleanup(self._close, fig)
        self.assertEqual(len(fig.axes), 1)
        self.assertIsNotNone(fig.axes[0].get_legend())


if __name__ == "__main__":
    unittest.main()
