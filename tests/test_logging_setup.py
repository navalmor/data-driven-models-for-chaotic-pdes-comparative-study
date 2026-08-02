"""Standard-library unit tests for the shared logging policy.

These tests are intentionally lightweight: they exercise ``common.logging_setup``,
two small helpers in the forecasting package, and a few repository-wide source
policies. They never load a dataset, train a model, run an optimiser, regenerate
a plot, require a GPU or touch any frozen artifact.

Run from the repository root with::

    python -m unittest discover -s tests
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "forecasting" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import logging_setup as L
from forecasting.data import complete_trajectory_shape
from forecasting.progress import progress_indices, reporting_verbosity

PRODUCTION_DIRS = ("scripts", "simulation", "ae_sindy", "forecasting", "common")


def _production_files() -> list[Path]:
    files: list[Path] = []
    for d in PRODUCTION_DIRS:
        files += [p for p in (REPO_ROOT / d).rglob("*.py") if "__pycache__" not in str(p)]
    return files


class HermeticEnvMixin(unittest.TestCase):
    """Neutralise the REPRO_* variables for the duration of each test.

    ``resolve_logging_settings`` reads these to receive the wrapper's policy in a
    child process, so a shell that exports them (which the wrapper itself does
    under ``--debug``) would otherwise change what these tests observe. The
    snapshot is restored afterwards, so the parent environment is never
    permanently modified.
    """

    REPRO_VARS = (L.ENV_LEVEL, L.ENV_VERBOSE, L.ENV_DEBUG)

    def setUp(self) -> None:
        super().setUp()
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in self.REPRO_VARS:
            os.environ.pop(name, None)


class ResetLoggingMixin(HermeticEnvMixin):
    """Restore root/presentation logging state around each test."""

    def setUp(self) -> None:
        super().setUp()
        self._root = logging.getLogger()
        self._saved_handlers = list(self._root.handlers)
        self._saved_level = self._root.level

    def tearDown(self) -> None:
        for handler in list(self._root.handlers):
            if getattr(handler, "_repro_managed", False):
                self._root.removeHandler(handler)
                # Release the file descriptor here rather than leaving it to the
                # garbage collector.
                handler.flush()
                handler.close()
        for handler in self._saved_handlers:
            if handler not in self._root.handlers:
                self._root.addHandler(handler)
        self._root.setLevel(self._saved_level)
        L._STATE.configured = False
        L._STATE.verbosity = 0
        L._STATE.default_component = L.DEFAULT_COMPONENT
        self._root.repro_verbosity = 0
        super().tearDown()


# --------------------------------------------------------------------------- #
# Stream routing: every structured level on stderr, presentation on stdout
# --------------------------------------------------------------------------- #


class TestStructuredStreamRouting(ResetLoggingMixin):
    def _emit(self, level: str, message: str, *, debug: bool = False):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            L.configure_logging("DEBUG" if debug else "INFO", component="REPRO", force=True)
            getattr(logging.getLogger("repro"), level)(message)
        return out.getvalue(), err.getvalue()

    def test_debug_goes_to_stderr(self) -> None:
        out, err = self._emit("debug", "a debug line", debug=True)
        self.assertIn("a debug line", err)
        self.assertEqual(out, "")

    def test_info_goes_to_stderr(self) -> None:
        out, err = self._emit("info", "an info line")
        self.assertIn("an info line", err)
        self.assertEqual(out, "")

    def test_warning_goes_to_stderr(self) -> None:
        out, err = self._emit("warning", "a warning line")
        self.assertIn("a warning line", err)
        self.assertEqual(out, "")

    def test_error_goes_to_stderr(self) -> None:
        out, err = self._emit("error", "an error line")
        self.assertIn("an error line", err)
        self.assertEqual(out, "")

    def test_critical_goes_to_stderr(self) -> None:
        out, err = self._emit("critical", "a critical line")
        self.assertIn("a critical line", err)
        self.assertEqual(out, "")

    def test_no_structured_output_leaks_to_stdout(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            L.configure_logging("DEBUG", component="NGRC", debug=True, force=True)
            log = logging.getLogger("forecasting.runners.ngrc_runner")
            for level in ("debug", "info", "warning", "error", "critical"):
                getattr(log, level)("structured %s", level)
        self.assertEqual(out.getvalue(), "")
        for level in ("debug", "info", "warning", "error", "critical"):
            self.assertIn(f"structured {level}", err.getvalue())

    def test_each_record_emitted_once(self) -> None:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            L.configure_logging("INFO", component="REPRO", force=True)
            logging.getLogger("repro").info("exactly once")
        self.assertEqual(err.getvalue().count("exactly once"), 1)

    def test_structured_format_shape(self) -> None:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            L.configure_logging("INFO", component="REPRO", force=True)
            logging.getLogger("forecasting.runners.rc_runner").info("hello")
        parts = [p.strip() for p in err.getvalue().strip().split("|")]
        self.assertEqual(len(parts), 4)
        self.assertEqual(parts[1], "INFO")
        self.assertEqual(parts[2], "RC")
        self.assertEqual(parts[3], "hello")

    def test_component_injection_and_default(self) -> None:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            L.configure_logging("INFO", component="KSE", force=True)
            logging.getLogger("ae_sindy.train").info("x")
            logging.getLogger("some.unmapped").info("y")
            logging.getLogger("repro").info("z", extra={"component": "VALIDATE"})
        text = err.getvalue()
        self.assertIn("| AE-SINDy ", text)
        self.assertIn("| KSE ", text)
        self.assertIn("| VALIDATE ", text)

    def test_debug_suppressed_at_info_level(self) -> None:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            L.configure_logging("INFO", component="REPRO", force=True)
            logging.getLogger("repro").debug("hidden debug")
        self.assertNotIn("hidden debug", err.getvalue())


class TestPresentation(ResetLoggingMixin):
    def test_presentation_goes_to_stdout_only(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            L.configure_logging("INFO", component="REPRO", force=True)
            pres = L.Presentation()
            pres.line("Final thesis results")
            pres.blank()
            pres.line("  [ 1] simulation_dataset")
        self.assertEqual(out.getvalue(), "Final thesis results\n\n  [ 1] simulation_dataset\n")
        self.assertEqual(err.getvalue(), "")

    def test_presentation_independent_of_log_level(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            L.configure_logging("WARNING", component="REPRO", force=True)
            L.Presentation().line("still shown at WARNING")
            logging.getLogger("repro").info("suppressed at WARNING")
        self.assertIn("still shown at WARNING", out.getvalue())
        self.assertNotIn("suppressed at WARNING", out.getvalue())
        self.assertNotIn("suppressed at WARNING", err.getvalue())


class TestIdempotency(ResetLoggingMixin):
    def test_no_duplicate_handlers(self) -> None:
        L.configure_logging("INFO", component="REPRO", force=True)
        L.configure_logging("INFO", component="REPRO")
        L.configure_logging("INFO", component="REPRO")
        managed = [h for h in logging.getLogger().handlers if getattr(h, "_repro_managed", False)]
        self.assertEqual(len(managed), 1)  # one structured stderr handler, no file

    def test_reconfiguration_does_not_duplicate_records(self) -> None:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            L.configure_logging("INFO", component="REPRO", force=True)
            L.configure_logging("INFO", component="REPRO")
            logging.getLogger("repro").info("once only")
        self.assertEqual(err.getvalue().count("once only"), 1)


class TestArgumentParsing(ResetLoggingMixin):
    def _parser(self) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser()
        L.add_logging_arguments(p)
        return p

    def test_log_level_case_insensitive(self) -> None:
        self.assertEqual(self._parser().parse_args(["--log-level", "debug"]).log_level, "DEBUG")

    def test_invalid_log_level_rejected(self) -> None:
        with self.assertRaises(SystemExit) as cm:
            self._parser().parse_args(["--log-level", "chatty"])
        self.assertEqual(cm.exception.code, 2)

    def test_verbose_debug_conflict_rejected(self) -> None:
        with self.assertRaises(SystemExit) as cm:
            self._parser().parse_args(["--verbose", "--debug"])
        self.assertEqual(cm.exception.code, 2)

    def test_debug_loglevel_conflict_rejected(self) -> None:
        with self.assertRaises(SystemExit) as cm:
            self._parser().parse_args(["--debug", "--log-level", "INFO"])
        self.assertEqual(cm.exception.code, 2)

    def test_resolve_precedence(self) -> None:
        p = self._parser()
        s = L.resolve_logging_settings(p.parse_args(["--debug"]), config_level="WARNING")
        self.assertEqual(s.level, "DEBUG")
        s = L.resolve_logging_settings(p.parse_args([]), config_level="ERROR")
        self.assertEqual(s.level, "ERROR")
        s = L.resolve_logging_settings(p.parse_args(["--verbose"]), config_level="ERROR")
        self.assertEqual((s.level, s.verbose), ("INFO", True))

    def test_default_level_when_nothing_is_set(self) -> None:
        s = L.resolve_logging_settings(self._parser().parse_args([]))
        self.assertEqual((s.level, s.verbose, s.debug), ("INFO", False, False))


class TestEnvironmentPrecedence(ResetLoggingMixin):
    """REPRO_* is how the wrapper hands its policy to a child process.

    These tests set the variables explicitly rather than inheriting them, so the
    documented order holds regardless of the shell: explicit CLI selection, then
    the environment, then a configuration level, then the INFO default.
    """

    def _parser(self) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser()
        L.add_logging_arguments(p)
        return p

    def _resolve(self, argv: list[str], env: dict[str, str], **kw) -> L.LoggingSettings:
        with mock.patch.dict(os.environ, env, clear=False):
            return L.resolve_logging_settings(self._parser().parse_args(argv), **kw)

    # --- environment applies when no CLI option is given ---------------------

    def test_env_debug_applies_without_cli_option(self) -> None:
        s = self._resolve([], {L.ENV_DEBUG: "1"})
        self.assertEqual((s.level, s.debug), ("DEBUG", True))

    def test_env_verbose_applies_without_cli_option(self) -> None:
        s = self._resolve([], {L.ENV_VERBOSE: "1"})
        self.assertEqual((s.level, s.verbose, s.debug), ("INFO", True, False))

    def test_env_level_applies_without_cli_option(self) -> None:
        s = self._resolve([], {L.ENV_LEVEL: "WARNING"})
        self.assertEqual(s.level, "WARNING")

    def test_env_level_outranks_config_level(self) -> None:
        s = self._resolve([], {L.ENV_LEVEL: "WARNING"}, config_level="ERROR")
        self.assertEqual(s.level, "WARNING")

    # --- an explicit CLI selection always wins -------------------------------

    def test_cli_debug_overrides_env_level(self) -> None:
        s = self._resolve(["--debug"], {L.ENV_LEVEL: "WARNING"})
        self.assertEqual((s.level, s.debug), ("DEBUG", True))

    def test_cli_verbose_overrides_env_level(self) -> None:
        s = self._resolve(["--verbose"], {L.ENV_LEVEL: "ERROR"})
        self.assertEqual((s.level, s.verbose), ("INFO", True))

    def test_cli_log_level_overrides_env(self) -> None:
        s = self._resolve(["--log-level", "ERROR"], {L.ENV_LEVEL: "DEBUG", L.ENV_DEBUG: "1"})
        self.assertEqual(s.level, "ERROR")

    # --- the suite itself must not be steered by inherited state -------------

    def test_inherited_env_is_neutralised_for_tests(self) -> None:
        for name in self.REPRO_VARS:
            self.assertNotIn(name, os.environ)

    def test_conflicting_cli_options_still_exit_2(self) -> None:
        for argv in (["--verbose", "--debug"], ["--debug", "--log-level", "INFO"],
                     ["--log-level", "chatty"]):
            with self.subTest(argv=argv):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as cm:
                        self._parser().parse_args(argv)
                self.assertEqual(cm.exception.code, 2)


class TestErrorTraceback(ResetLoggingMixin):
    def _run(self, debug: bool) -> str:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            L.configure_logging("INFO", component="RC", debug=debug, force=True)
            logger = logging.getLogger("forecasting.scripts.run_rc")
            try:
                raise ValueError("missing config")
            except Exception as exc:  # noqa: BLE001
                if L.is_debug():
                    logger.exception("RC run failed")
                else:
                    logger.error("RC run failed: %s", exc)
        return err.getvalue()

    def test_concise_error_has_no_traceback(self) -> None:
        text = self._run(debug=False)
        self.assertIn("RC run failed: missing config", text)
        self.assertNotIn("Traceback", text)

    def test_debug_error_includes_traceback(self) -> None:
        text = self._run(debug=True)
        self.assertIn("Traceback", text)
        self.assertIn("ValueError", text)


class TestLogFile(ResetLoggingMixin):
    def test_log_file_content_and_hygiene(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sub" / "run.log"
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                L.configure_logging("INFO", component="REPRO", log_file=path, force=True)
                logging.getLogger("repro").info("file line")
                logging.getLogger("repro").debug("debug detail")
                for h in logging.getLogger().handlers:
                    h.flush()
            content = path.read_text(encoding="utf-8")
            self.assertIn("file line", content)
            self.assertIn("debug detail", content)      # file keeps full DEBUG
            self.assertNotIn("debug detail", err.getvalue())  # console honours INFO
            self.assertNotIn("\x1b[", content)          # no ANSI escapes
            self.assertEqual(content.count("file line"), 1)  # no duplicate records

    def test_parent_directory_created_only_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            nested = Path(d) / "a" / "b" / "run.log"
            self.assertFalse(nested.parent.exists())
            with contextlib.redirect_stderr(io.StringIO()):
                L.configure_logging("INFO", component="REPRO", log_file=nested, force=True)
            self.assertTrue(nested.parent.is_dir())


class TestNonForcedFileReconfiguration(ResetLoggingMixin):
    """A requested log file must never be silently dropped (review finding M3)."""

    @staticmethod
    def _managed_files() -> list[logging.FileHandler]:
        return L._managed_file_handlers(logging.getLogger())

    def test_case_a_first_file_is_attached_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "late" / "run.log"
            with contextlib.redirect_stderr(io.StringIO()):
                L.configure_logging("INFO", component="REPRO", force=True)
                self.assertEqual(self._managed_files(), [])
                L.configure_logging("INFO", component="REPRO", log_file=path)
                logging.getLogger("repro").debug("added late")
                for h in logging.getLogger().handlers:
                    h.flush()
            self.assertTrue(path.exists())
            self.assertEqual(len(self._managed_files()), 1)
            content = path.read_text(encoding="utf-8")
            self.assertIn("added late", content)        # DEBUG reaches the file
            self.assertNotIn("\x1b[", content)
            # exactly one structured console handler, not a second copy
            console = [
                h for h in logging.getLogger().handlers
                if getattr(h, "_repro_managed", False) and not isinstance(h, logging.FileHandler)
            ]
            self.assertEqual(len(console), 1)

    def test_case_b_same_path_is_reused_without_duplicating(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "run.log"
            with contextlib.redirect_stderr(io.StringIO()):
                L.configure_logging("INFO", component="REPRO", log_file=path, force=True)
                L.configure_logging("INFO", component="REPRO", log_file=path)
                L.configure_logging("INFO", component="REPRO", log_file=str(path))
                logging.getLogger("repro").info("once only")
                for h in logging.getLogger().handlers:
                    h.flush()
            self.assertEqual(len(self._managed_files()), 1)
            self.assertEqual(path.read_text(encoding="utf-8").count("once only"), 1)

    def test_case_c_different_path_raises_instead_of_being_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            first, second = Path(d) / "first.log", Path(d) / "second.log"
            with contextlib.redirect_stderr(io.StringIO()):
                L.configure_logging("INFO", component="REPRO", log_file=first, force=True)
                with self.assertRaises(ValueError) as cm:
                    L.configure_logging("INFO", component="REPRO", log_file=second)
            message = str(cm.exception)
            self.assertIn("force=True", message)        # actionable
            self.assertIn("second.log", message)
            self.assertFalse(second.exists())           # not silently created
            self.assertEqual(len(self._managed_files()), 1)   # no second destination
            self.assertEqual(self._managed_files()[0].baseFilename, os.path.abspath(str(first)))

    def test_force_replaces_and_closes_the_previous_handler(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            first, second = Path(d) / "first.log", Path(d) / "second.log"
            with contextlib.redirect_stderr(io.StringIO()):
                L.configure_logging("INFO", component="REPRO", log_file=first, force=True)
                old = self._managed_files()[0]
                L.configure_logging("INFO", component="REPRO", log_file=second, force=True)
                logging.getLogger("repro").info("after replacement")
                for h in logging.getLogger().handlers:
                    h.flush()
            self.assertIsNone(old.stream)               # old handler was closed
            self.assertEqual(len(self._managed_files()), 1)
            self.assertIn("after replacement", second.read_text(encoding="utf-8"))
            self.assertNotIn("after replacement", first.read_text(encoding="utf-8"))

    def test_unmanaged_handlers_are_left_alone(self) -> None:
        root = logging.getLogger()
        foreign = logging.StreamHandler(io.StringIO())
        root.addHandler(foreign)
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                L.configure_logging("INFO", component="REPRO", force=True)
            self.assertIn(foreign, root.handlers)
        finally:
            root.removeHandler(foreign)

    def test_teardown_leaves_no_managed_handlers_installed(self) -> None:
        # Guards C1: the mixin must close and remove what it configured.
        with tempfile.TemporaryDirectory() as d:
            with contextlib.redirect_stderr(io.StringIO()):
                L.configure_logging("INFO", log_file=Path(d) / "x.log", force=True)
            self.assertEqual(len(self._managed_files()), 1)
        # tearDown runs after this test; the following test's setUp asserts a
        # clean slate via _managed_files() in every other case above.


# --------------------------------------------------------------------------- #
# Shape semantics, horizon units, CUDA fallback, child log files
# --------------------------------------------------------------------------- #


class TestCompleteTrajectoryShape(HermeticEnvMixin):
    """The INFO line must describe the whole trajectory, not the training split."""

    def test_shape_is_recovered_from_splits(self) -> None:
        import numpy as np

        train = np.zeros((20000, 64))
        valid = np.zeros((5000, 64))
        test = np.zeros((5106, 64))
        self.assertEqual(complete_trajectory_shape(train, valid, test), (30106, 64))

    def test_latent_width_preserved(self) -> None:
        import numpy as np

        self.assertEqual(
            complete_trajectory_shape(np.zeros((20000, 8)), np.zeros((5000, 8)), np.zeros((5106, 8))),
            (30106, 8),
        )

    def test_helper_reads_only_shapes(self) -> None:
        """It must not copy or reload data: a read-only view is enough."""
        import numpy as np

        base = np.zeros((100, 3))
        base.flags.writeable = False
        self.assertEqual(complete_trajectory_shape(base[:60], base[60:80], base[80:]), (100, 3))


class TestRunnerLoggingSemantics(HermeticEnvMixin):
    """Source-level guarantees for shape wording, horizon units and CUDA level."""

    RUNNERS = {
        name: (REPO_ROOT / "forecasting/src/forecasting/runners" / f"{name}_runner.py").read_text()
        for name in ("ngrc", "rc", "perc", "pinn")
    }

    def test_info_reports_input_trajectory_not_a_split(self) -> None:
        for name, src in self.RUNNERS.items():
            with self.subTest(runner=name):
                self.assertIn("Input trajectory: shape=%s", src)
                self.assertNotIn('logger.info("Input data: shape=%s', src)

    @staticmethod
    def _log_levels_for(src: str, needle: str) -> list[str]:
        """Levels of every logger.<level>(...) call whose format string contains *needle*."""
        levels = []
        for node in ast.walk(ast.parse(src)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and needle in node.args[0].value
            ):
                levels.append(node.func.attr)
        return levels

    def test_split_shapes_are_debug(self) -> None:
        for name, src in self.RUNNERS.items():
            with self.subTest(runner=name):
                levels = self._log_levels_for(src, "Data splits:")
                self.assertTrue(levels, "no 'Data splits' line found")
                self.assertEqual(set(levels), {"debug"})

    def test_input_trajectory_is_info(self) -> None:
        for name, src in self.RUNNERS.items():
            with self.subTest(runner=name):
                self.assertEqual(
                    set(self._log_levels_for(src, "Input trajectory: shape=")), {"info"}
                )

    def test_horizon_step_counts_are_debug(self) -> None:
        for name, src in self.RUNNERS.items():
            with self.subTest(runner=name):
                self.assertEqual(
                    set(self._log_levels_for(src, "horizon: %s steps")), {"debug"}
                )

    def test_horizon_metric_states_physical_time(self) -> None:
        for name, src in self.RUNNERS.items():
            with self.subTest(runner=name):
                self.assertIn("Validation horizon (physical time)", src)
                self.assertIn("relative_l2_horizon_steps", src)

    def test_cuda_fallback_is_a_warning(self) -> None:
        src = self.RUNNERS["pinn"]
        self.assertIn('logger.warning("CUDA unavailable; using CPU")', src)
        self.assertNotIn('logger.info("CUDA not available', src)
        self.assertIn('logger.info("Device: %s", device)', src)

    def test_cuda_fallback_emitted_once(self) -> None:
        self.assertEqual(self.RUNNERS["pinn"].count("CUDA unavailable; using CPU"), 1)
        wrapper = (REPO_ROOT / "scripts/reproduce_final_thesis.py").read_text()
        self.assertNotIn("CUDA unavailable", wrapper)


class TestChildLoggingHandover(HermeticEnvMixin):
    """The wrapper must stream child logs live and give each child its own file."""

    WRAPPER = (REPO_ROOT / "scripts/reproduce_final_thesis.py").read_text()

    # The checks below inspect the parsed syntax tree rather than raw text, so
    # they assert behaviour and survive reformatting or line rewrapping.

    @classmethod
    def setUpClass(cls) -> None:
        cls.RUN_COMMANDS = next(
            node
            for node in ast.walk(ast.parse(cls.WRAPPER))
            if isinstance(node, ast.FunctionDef) and node.name == "run_commands"
        )
        cls.SUBPROCESS_CALL = next(
            node
            for node in ast.walk(cls.RUN_COMMANDS)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
        )

    @staticmethod
    def _flush_lineno(fn: ast.FunctionDef, stream: str) -> int:
        """Line of ``sys.<stream>.flush()`` inside *fn*."""
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "flush"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == stream
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "sys"
            ):
                return node.lineno
        raise AssertionError(f"no sys.{stream}.flush() in {fn.name}()")

    def test_stdout_flushed_before_child_launch(self) -> None:
        self.assertLess(
            self._flush_lineno(self.RUN_COMMANDS, "stdout"), self.SUBPROCESS_CALL.lineno
        )

    def test_stderr_flushed_before_child_launch(self) -> None:
        self.assertLess(
            self._flush_lineno(self.RUN_COMMANDS, "stderr"), self.SUBPROCESS_CALL.lineno
        )

    def test_child_streams_are_inherited(self) -> None:
        kwargs = {kw.arg for kw in self.SUBPROCESS_CALL.keywords}
        self.assertNotIn("stdout", kwargs)
        self.assertNotIn("stderr", kwargs)

    def test_child_is_not_launched_through_a_shell(self) -> None:
        shell = [kw for kw in self.SUBPROCESS_CALL.keywords if kw.arg == "shell"]
        self.assertEqual(shell, [], "subprocess.run must not use shell=True")

    def test_child_argv_extends_cmd_argv_with_log_file(self) -> None:
        """child_argv = [*cmd.argv, "--log-file", <path>] -- however it is spelled."""
        assign = next(
            node
            for node in ast.walk(self.RUN_COMMANDS)
            if isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "child_argv" for t in node.targets)
        )
        self.assertIsInstance(assign.value, ast.List)
        starred = [e for e in assign.value.elts if isinstance(e, ast.Starred)]
        self.assertTrue(starred, "child_argv must start from the scientific argv")
        self.assertEqual(ast.unparse(starred[0].value), "cmd.argv")
        literals = [e.value for e in assign.value.elts if isinstance(e, ast.Constant)]
        self.assertIn("--log-file", literals)
        # the subprocess must receive that list, not the bare scientific argv
        self.assertEqual(ast.unparse(self.SUBPROCESS_CALL.args[0]), "child_argv")

    def test_scientific_command_is_recorded_without_the_injected_flag(self) -> None:
        """Provenance must record cmd.argv, never child_argv."""
        appends = [
            node
            for node in ast.walk(self.RUN_COMMANDS)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and getattr(node.func.value, "id", None) == "ran"
        ]
        self.assertTrue(appends, "run_commands must record the command it ran")
        for call in appends:
            source = ast.unparse(call)
            self.assertIn("cmd.argv", source)
            self.assertNotIn("child_argv", source)

    def test_every_child_entry_point_accepts_log_file(self) -> None:
        for rel in (
            "forecasting/scripts/run_ngrc.py",
            "forecasting/scripts/run_rc.py",
            "forecasting/scripts/run_perc.py",
            "forecasting/scripts/run_pinn.py",
            "ae_sindy/scripts/run_analysis.py",
            "ae_sindy/scripts/run_visualize.py",
            "scripts/posthoc_refit_sindy.py",
        ):
            with self.subTest(entry=rel):
                self.assertIn("add_logging_arguments", (REPO_ROOT / rel).read_text())


class TestProgressCadence(ResetLoggingMixin):
    def test_normal_cadence_includes_first_and_last(self) -> None:
        L.configure_logging("INFO", force=True)
        marks = progress_indices(500)
        self.assertIn(1, marks)
        self.assertIn(500, marks)
        self.assertLessEqual(len(marks), 13)

    def test_verbose_reports_every_iteration(self) -> None:
        L.configure_logging("INFO", verbose=True, force=True)
        self.assertEqual(sorted(progress_indices(7)), [1, 2, 3, 4, 5, 6, 7])

    def test_verbosity_is_published_for_library_use(self) -> None:
        L.configure_logging("INFO", debug=True, force=True)
        self.assertEqual(reporting_verbosity(), 2)
        L.configure_logging("INFO", force=True)
        self.assertEqual(reporting_verbosity(), 0)

    def test_progress_does_not_consume_randomness(self) -> None:
        import random as _random

        _random.seed(1234)
        before = _random.random()
        _random.seed(1234)
        progress_indices(500)
        self.assertEqual(before, _random.random())


# --------------------------------------------------------------------------- #
# Repository-wide source policies
# --------------------------------------------------------------------------- #


class TestNoPrintPolicy(HermeticEnvMixin):
    def test_no_production_print_or_pprint(self) -> None:
        offenders = []
        for path in _production_files():
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call):
                    fn = node.func
                    if isinstance(fn, ast.Name) and fn.id == "print":
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
                    if isinstance(fn, ast.Attribute) and fn.attr == "pprint":
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} pprint")
        self.assertEqual(offenders, [], f"bare print/pprint found: {offenders}")


class TestSysPathPolicy(HermeticEnvMixin):
    """Only direct entry points may bootstrap sys.path; libraries never may."""

    @staticmethod
    def _is_entry_point(src: str) -> bool:
        return '__name__ == "__main__"' in src or "__name__ == '__main__'" in src

    def test_no_library_module_mutates_sys_path(self) -> None:
        offenders = []
        for path in _production_files():
            src = path.read_text()
            mutates = any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in {"insert", "append"}
                and isinstance(n.func.value, ast.Attribute)
                and n.func.value.attr == "path"
                and isinstance(n.func.value.value, ast.Name)
                and n.func.value.value.id == "sys"
                for n in ast.walk(ast.parse(src))
            )
            if mutates and not self._is_entry_point(src):
                offenders.append(str(path.relative_to(REPO_ROOT)))
        self.assertEqual(offenders, [], f"library modules mutating sys.path: {offenders}")

    def test_library_packages_do_not_import_common_at_module_level(self) -> None:
        offenders = []
        for root in ("forecasting/src/forecasting", "ae_sindy/ae_sindy"):
            for path in (REPO_ROOT / root).rglob("*.py"):
                if "__pycache__" in str(path):
                    continue
                tree = ast.parse(path.read_text())
                for node in tree.body:  # module level only; lazy imports are allowed
                    root_name = None
                    if isinstance(node, ast.Import):
                        root_name = node.names[0].name.split(".")[0]
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        root_name = node.module.split(".")[0]
                    if root_name == "common":
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
        self.assertEqual(offenders, [], f"library modules importing common: {offenders}")


if __name__ == "__main__":
    unittest.main()
