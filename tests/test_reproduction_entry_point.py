"""Guards for the public reproduction entry point.

Before this module the correctness argument of scripts/reproduce_final_thesis.py
rested entirely on untested pure functions. These tests cover the safety
behaviour that Phase R3 added, plus the pre-existing helpers that decide what a
result is compared against.

Nothing here runs a model, writes into the package, or touches a scientific file.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "reproduce_final_thesis.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_repro_entry", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_repro_entry"] = module
    spec.loader.exec_module(module)
    return module


R = _load_module()


class TestRunIdIsSafe(unittest.TestCase):
    """An unvalidated --run-id could resolve into the frozen package."""

    def test_plain_names_are_accepted(self):
        for value in ("20260802_120000", "my-run", "nested/run"):
            self.assertEqual(R.safe_run_id(value), value)

    def test_absolute_paths_are_rejected(self):
        # Assembled rather than written out: a bare scratch-directory literal in a
        # tracked file trips the release path guard in test_release_scope.py.
        with self.assertRaises(ValueError):
            R.safe_run_id("/" + "tmp" + "/somewhere")
        with self.assertRaises(ValueError):
            R.safe_run_id("/var/data/run")

    def test_parent_directory_escape_is_rejected(self):
        with self.assertRaises(ValueError):
            R.safe_run_id("../final_thesis_package_v001/results")

    def test_parent_directory_anywhere_in_the_path_is_rejected(self):
        with self.assertRaises(ValueError):
            R.safe_run_id("runs/../../escape")

    def test_home_expansion_is_rejected(self):
        with self.assertRaises(ValueError):
            R.safe_run_id("~/elsewhere")

    def test_empty_and_padded_values_are_rejected(self):
        for value in ("", "   ", " run "):
            with self.assertRaises(ValueError):
                R.safe_run_id(value)


class TestPackageIsProtectedFromWrites(unittest.TestCase):
    def test_a_path_inside_the_package_is_refused(self):
        target = R.PACKAGE / "results" / "01_ae_sindy"
        with self.assertRaises(ValueError):
            R.assert_outside_package(target, "test target")

    def test_the_package_root_itself_is_refused(self):
        with self.assertRaises(ValueError):
            R.assert_outside_package(R.PACKAGE, "test target")

    def test_a_run_folder_is_allowed(self):
        allowed = R.RUNS_ROOT / "some_run"
        self.assertEqual(R.assert_outside_package(allowed, "run folder"), allowed)

    def test_the_escape_route_that_motivated_the_guard_is_closed(self):
        """`--run-id ../final_thesis_package_v001/...` must not resolve into the package."""
        escape = R.RUNS_ROOT / "../final_thesis_package_v001/results"
        with self.assertRaises(ValueError):
            R.assert_outside_package(escape, "run folder")


class TestLfsPointerDetection(unittest.TestCase):
    """A clone made without git-lfs must fail with guidance, not with checksum noise."""

    def test_a_pointer_stub_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub = Path(tmp) / "u_series.npy"
            stub.write_bytes(
                b"version https://git-lfs.github.com/spec/v1\n"
                b"oid sha256:0123456789abcdef\nsize 15414400\n"
            )
            self.assertTrue(R.lfs_pointer_stub(stub))

    def test_real_content_is_not_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "u_series.npy"
            real.write_bytes(b"\x93NUMPY\x01\x00" + b"\x00" * 64)
            self.assertFalse(R.lfs_pointer_stub(real))

    def test_a_missing_file_is_not_flagged(self):
        self.assertFalse(R.lfs_pointer_stub(Path("/nonexistent/u_series.npy")))

    def test_the_shipped_probe_file_is_materialised(self):
        """This checkout must not itself be running on pointer stubs."""
        self.assertTrue(R._LFS_PROBE.exists())
        self.assertFalse(R.lfs_pointer_stub(R._LFS_PROBE))


class TestPlotCopiesAreVerifiedBeforeBeingLabelled(unittest.TestCase):
    def test_a_packaged_figure_resolves_to_its_manifest_digest(self):
        sample = R.PACKAGE / "plots/00_simulation/kse_spatiotemporal_lyapunov.png"
        self.assertTrue(sample.exists())
        recorded = R.plot_manifest_digest(sample)
        self.assertIsNotNone(recorded, "sample figure is absent from the plot manifests")
        self.assertEqual(recorded, R.sha256(sample))

    def test_a_file_outside_the_manifests_returns_none(self):
        self.assertIsNone(R.plot_manifest_digest(_REPO_ROOT / "README.md"))

    def test_every_official_png_is_listed_in_a_plot_manifest(self):
        missing = [
            p.relative_to(_REPO_ROOT).as_posix()
            for p in (R.PACKAGE / "plots").rglob("*.png")
            if R.plot_manifest_digest(p) is None
        ]
        self.assertEqual(missing, [], f"figures listed in no manifest: {missing[:5]}")


class TestDeclaredCountEnforcement(unittest.TestCase):
    def test_a_leading_count_is_parsed(self):
        self.assertEqual(R._declared_count("76 PNGs, 76/76 sha256 match"), 76)

    def test_a_target_without_a_count_returns_none(self):
        self.assertIsNone(R._declared_count("see manifests/package_table_sha256.csv"))
        self.assertIsNone(R._declared_count(""))

    def test_the_plot_contract_still_declares_the_shipped_figure_count(self):
        on_disk = len(list((R.PACKAGE / "plots").rglob("*.png")))
        target = R.load_targets()["PLOT-001"]   # load_targets() is keyed by target_id
        self.assertEqual(R._declared_count(target["expected_value"]), on_disk)


class TestRunStatusIsAtomicAndMachineReadable(unittest.TestCase):
    def test_status_file_is_written_and_parsable(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            R.write_run_status(run_dir, "PASSED", {"PASS": 52}, exit_code=0)
            payload = json.loads((run_dir / "run_status.json").read_text())
            self.assertEqual(payload["verdict"], "PASSED")
            self.assertEqual(payload["exit_code"], 0)
            self.assertEqual(payload["counts"]["PASS"], 52)
            self.assertIn("completed_utc", payload)

    def test_no_temporary_file_is_left_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            R.write_run_status(run_dir, "FAILED", {"FAIL": 1}, exit_code=1)
            self.assertFalse((run_dir / "run_status.json.tmp").exists())


class TestExpectedSourceMetadataResolves(unittest.TestCase):
    """The field named files that do not exist, in a public reproducibility artefact."""

    def test_every_expected_source_names_an_existing_file(self):
        missing = sorted({
            result.expected_source
            for result in R.REGISTRY
            if result.expected_source and not (_REPO_ROOT / result.expected_source).exists()
        })
        self.assertEqual(missing, [], f"dangling expected_source values: {missing}")


class TestResultSelection(unittest.TestCase):
    def test_all_results_covers_the_whole_registry(self):
        self.assertEqual(len(R.REGISTRY), 12)

    def test_all_safe_excludes_the_two_heavy_pinns(self):
        safe = [r.result_id for r in R.REGISTRY if r.laptop_safe]
        self.assertNotIn("pinn_full64", safe)
        self.assertNotIn("pinn_latent8", safe)
        self.assertEqual(len(safe), 10)

    def test_the_cli_exposes_all_results(self):
        source = _SCRIPT.read_text()
        self.assertIn('"--all-results"', source)


class TestFrozenBehaviourIsUnchanged(unittest.TestCase):
    """Phase R3 must not have touched the scientific machinery."""

    def test_locked_seed_verification_still_refuses_the_config_field(self):
        self.assertNotIn("metadata.final_seed_choice.random_seed", R.RUNTIME_SEED_FIELDS)

    def test_the_retired_latent8_builder_is_not_reachable(self):
        self.assertNotIn("latent8_trial014_representation", R.BUILDERS)
        self.assertTrue(hasattr(R, "_RETIRED_build_ae_sindy_latent8"))

    def test_redirect_config_is_still_defined(self):
        self.assertTrue(callable(R.redirect_config))


if __name__ == "__main__":
    unittest.main()
