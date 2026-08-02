# Validation

The repository ships a validation contract that checks reproduced and frozen results against
recorded expectations. The reproduction wrapper,
[`scripts/reproduce_final_thesis.py`](../scripts/reproduce_final_thesis.py), reads this contract
and reports a pass/fail summary.

## The contract

The contract is four files under [`repro_validation/`](../repro_validation/):

- `expected/validation_targets.csv` — the list of validation targets: one row per checked value
  or artifact, with the expected value, the comparison method, and the tolerance to apply.
- `expected/simulation_expected.json` — expected properties of the frozen simulation dataset.
- `manifests/package_table_sha256.csv` — checksums of the published tables and plot index.
- `tolerances/tolerance_policy.json` — the numeric tolerances the comparisons use.

Targets are compared in one of a few ways: exact checksums for files that must be byte-identical,
numeric comparisons within a stated tolerance for reproduced metrics, and exact integer or field
comparisons for discrete values such as seeds and dimensions.

## Two kinds of run

**Validate the existing package.** This checks the shipped package against the contract without
running any models:

```bash
python scripts/reproduce_final_thesis.py --validate-existing --yes
```

It runs no models. It evaluates all 54 targets and reports **52 PASS, 0 FAIL, 0 REVIEW_REQUIRED, 2 SKIPPED, 0 ERROR**, exiting `0`. A failure exits `1`; an unresolved target exits `3`.

**Reproduce and validate.** This **retrains and re-evaluates** the locked configurations, then
validates the fresh output. It evaluates 52 targets and reports **50 PASS, 0 FAIL, 0 REVIEW,
2 SKIPPED, 0 ERROR** — two fewer than route 1 because `PLOT-001` and `TBL-001` are package-wide
checks that a per-result run does not evaluate.

```bash
python scripts/reproduce_final_thesis.py --list             # the 12 final results
python scripts/reproduce_final_thesis.py --results <id> --yes
python scripts/reproduce_final_thesis.py --all-results --yes    # all 12, about 30 minutes
```

The two runs differ by two targets: `PLOT-001` (the plot index) and `TBL-001` (the tables) are
package-wide checks, not tied to any single reproduced result, so they are part of the
existing-package validation but not of a fresh per-result run.

Both commands accept the shared logging options (`--verbose`, `--debug`, `--log-level`,
`--log-file`). The formal summary above is plain text on stdout and is printed regardless of the
level, while all structured diagnostics go to stderr, so `> summary.txt 2> run.log` separates the
two cleanly. During a reproduction each model's component logs appear live on stderr beneath its
workflow heading and are also written to a per-component file under
`reproduction_runs/<run-id>/logs/`. See the [README](../README.md#logging-and-verbosity) for
details. Logging never affects the validation outcome.

## The two skipped targets

Two targets are always skipped: `FC-01S` and `FC-02S`, the seed checks for `ngrc_full64` and
`ngrc_latent8`. NGRC is deterministic and has no random seed, so a seed comparison is undefined
for it. These are skipped by definition, not because of unfinished work.

## Reading the status labels

- **PASS** — the target matched its expected value within tolerance.
- **FAIL** — the target did not match; this should not occur for a clean checkout and run.
- **REVIEW** — the result needs manual inspection; not expected in the default workflow.
- **SKIPPED** — the target does not apply (the two NGRC seed checks).
- **ERROR** — the check could not be carried out (for example a missing input).

## Package integrity

Alongside the target contract, the package carries SHA-256 manifests covering its files. Two
of them are swept by the contract itself: `PLOT-001` checks every figure listed in
`stage6_final_plots_sha256_manifest.csv` and `stage7b_seed_robust_plots_sha256_manifest.csv`,
and `TBL-001` checks the tables. Integrity confirms the package is intact; the contract
confirms the values are correct.

The remaining manifests are not swept automatically. To check every manifest in the
repository, recompute the digests and compare them against each `*_manifest.csv`.

## Scope of the guarantee

The contract confirms that the reproduction matches the recorded results within the stated
tolerances on a compatible environment. It does not promise bitwise-identical output on
arbitrary hardware. The chaotic simulation is validated by checksum rather than regenerated, and
the GPU-trained autoencoder representation is validated by checksum rather than retrained; see
[simulation.md](simulation.md) and [ae_sindy.md](ae_sindy.md).
