"""Release-scope guards.

Two properties must hold for every commit on the public release branch, and both
are cheap enough to assert on every test run:

1.  No tracked file leaks a private or machine-specific filesystem path.
2.  No tracked path belongs to the private thesis workspace.

Both guards are deliberately narrow. The allow-lists match an exact line, not a
file and not a directory, so a new violation in an already-allow-listed file
still fails.
"""
from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Tokens that must not reach a public release. "/apps/" is included because the
# conda tree it names is specific to one HPC site, not because it is secret.
FORBIDDEN_TOKENS = (
    "/home/",
    "/scratch/",
    "/tmp/",
    "/apps/",
    "DataDrivenModelsOfChaoticPDEs",
    "dsnf120h",
)

# Files whose *purpose* is to define a forbidden-token list. They necessarily
# contain the tokens as data. The set is asserted to be exactly this, so a new
# file cannot quietly join it and smuggle a real path through.
GUARD_DEFINITION_FILES = frozenset({
    "tests/test_release_scope.py",
    "tests/test_optimizer_public_dataset.py",
    "tests/test_plotting_conventions.py",
    "forecasting/scripts/extract_optimizer_figure_data.py",
})

# The only site-specific paths permitted in the release, matched by exact line.
# Both are documented interpreter defaults, both are existence-guarded, and both
# are overridable by --system-python / --torch-python or the REPRO_SYSTEM_PYTHON
# / REPRO_TORCH_PYTHON environment variables (see docs/environment.md).
SITE_PATH_ALLOWLIST = {
    (
        "scripts/reproduce_final_thesis.py",
        '_HPC_SYS_PY = "/usr/bin/python3"',
    ),
    (
        "scripts/reproduce_final_thesis.py",
        '_HPC_CONDA_PY = "/apps/python/3.12-conda/envs/pytorch2.6-py3.12/bin/python"',
    ),
}


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=_REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line.strip()]


def _is_text(raw: bytes) -> bool:
    if b"\x00" in raw[:8192]:
        return False
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def scan_line_for_private_paths(rel_path: str, line: str) -> list[str]:
    """Tokens that make this line a violation. Empty means the line is clean.

    Exposed so the detector itself can be tested against synthetic input,
    which is what proves the guard would fail on a real violation.
    """
    if rel_path in GUARD_DEFINITION_FILES:
        return []
    stripped = line.strip()
    if (rel_path, stripped) in SITE_PATH_ALLOWLIST:
        return []
    return [t for t in FORBIDDEN_TOKENS if t in line]


class TestNoPrivatePathsInTrackedFiles(unittest.TestCase):
    """Sweeps every tracked text file, not only the optimiser dataset."""

    def test_detector_catches_a_private_path(self):
        """The guard must fail on a real violation, not merely pass on clean input."""
        self.assertEqual(
            scan_line_for_private_paths("docs/example.md", "see /home/hpc/dsnf120h/x"),
            ["/home/", "dsnf120h"],
        )
        self.assertEqual(
            scan_line_for_private_paths("common/plotting.py", 'p = "/scratch/tmp"'),
            ["/scratch/"],
        )

    def test_detector_respects_only_the_exact_allow_listed_line(self):
        """An allow-listed file must still fail on a different private path."""
        self.assertEqual(
            scan_line_for_private_paths(
                "scripts/reproduce_final_thesis.py",
                '_HPC_SYS_PY = "/usr/bin/python3"',
            ),
            [],
        )
        self.assertEqual(
            scan_line_for_private_paths(
                "scripts/reproduce_final_thesis.py",
                'leaked = "/home/hpc/dsnf120h/secret"',
            ),
            ["/home/", "dsnf120h"],
        )

    def test_guard_definition_files_all_exist(self):
        """Prevents a new file from joining the exemption list unnoticed.

        Existence is checked on disk rather than against `git ls-files`, so the
        guard is meaningful before its own first commit as well as after.
        """
        for rel in GUARD_DEFINITION_FILES:
            self.assertTrue(
                (_REPO_ROOT / rel).is_file(), f"guard-definition file missing: {rel}"
            )

    def test_no_tracked_text_file_contains_a_private_path(self):
        violations: list[str] = []
        for rel in _tracked_files():
            path = _REPO_ROOT / rel
            if not path.is_file():
                continue
            raw = path.read_bytes()
            if not _is_text(raw):
                continue
            for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
                found = scan_line_for_private_paths(rel, line)
                if found:
                    violations.append(f"{rel}:{number} {found} :: {line.strip()[:100]}")
        self.assertEqual(violations, [], "private paths in tracked files:\n" + "\n".join(violations))


class TestThesisTreeIsNotTracked(unittest.TestCase):
    """The thesis workspace is private and must never enter the release."""

    def test_detector_catches_a_thesis_path(self):
        paths = ["README.md", "thesis/overleaf_project/MastersThesis.tex"]
        self.assertEqual(
            [p for p in paths if p.startswith("thesis/")],
            ["thesis/overleaf_project/MastersThesis.tex"],
        )

    def test_no_tracked_path_begins_with_thesis(self):
        offenders = [p for p in _tracked_files() if p.startswith("thesis/")]
        self.assertEqual(offenders, [], f"thesis paths tracked in the release: {offenders[:10]}")

    def test_final_thesis_package_is_not_caught_by_the_guard(self):
        """The package shares a name fragment but is authoritative release content."""
        tracked = _tracked_files()
        self.assertTrue(any(p.startswith("final_thesis_package_v001/") for p in tracked))
        self.assertEqual([p for p in tracked if p.startswith("thesis/")], [])


class TestInternalReviewWorkspaceIsIgnored(unittest.TestCase):
    """_local_review/ holds internal audit material and must never be committable."""

    def test_local_review_is_ignored_by_the_tracked_gitignore(self):
        """The rule must live in the committed .gitignore.

        `git check-ignore` also honours .git/info/exclude, which is local and
        untracked: a rule there protects this checkout but not a fresh clone or
        any other contributor. Asserting the *source* of the matching rule is
        what makes this guard mean something for the release.
        """
        result = subprocess.run(
            ["git", "check-ignore", "-v", "_local_review/"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode, 0, "_local_review/ is not ignored at all"
        )
        source = result.stdout.split(":", 1)[0]
        self.assertTrue(
            source.endswith(".gitignore"),
            f"_local_review/ is ignored by {source!r}, not by the tracked .gitignore; "
            "a fresh clone would not be protected",
        )

    def test_local_review_is_not_tracked(self):
        offenders = [p for p in _tracked_files() if p.startswith("_local_review/")]
        self.assertEqual(offenders, [], f"internal review files tracked: {offenders[:5]}")


if __name__ == "__main__":
    unittest.main()
