# Portable, De-identified Aggregate Reproduction Draft

## Material Passport

Date: 2026-09-22. Fresh packaging, extraction, isolated execution, numerical
cross-checks and regression tests. The sixteen underlying analyses and their
intervals are cached-verified, not new experiments. No new training, forecast
readout, bootstrap, calibration, confirmation, deployment or submission.

The preceding user-facing status answer was not research progress. This
continuation adds an executable artifact that did not previously exist, rather
than revising the manuscript again. Scientific readiness is still false.

## Delivered Artifact

Local-only archive:
`data/stage_cvpr2027_experiments/blinded_reproduction_v1/evidence_bundle.zip`.

- 305,464 bytes; thirteen explicitly allowlisted files.
- A standalone Python standard-library entrypoint; no repository imports.
- Selected aggregate evidence fields, not raw trajectories or formatted-table
  inputs; eight policies, 32 site rows, 96 site/seed rows, fourteen contrasts and
  three joint-support summaries.
- Seven expected result files and two previously verified reference SVGs.
- An integrity manifest and usage/limitations document with no direct author
  names, personal paths, institution or project-repository links.

Archive SHA256:
`0ab7306e051a0f1d62a9f071c92146f5341dd994ef457017ee435e75384a3b7d`.

The fixed source-to-export comparison checks 1,062 fields, including negative
contrasts, worst site/seed easy degradation, incomplete and unknown outcomes,
and failed primary comparisons. Changing a primary failure into a positive
claim or a joint-support diagnostic into predictive success is rejected.

## Actual Execution

The archive was extracted to a fresh temporary directory. Python 3.11.1 ran
with `-I -S -B`, a temporary home, disabled site packages and a restricted
environment. Runtime audit hooks refused network/subprocess use and file access
outside the extracted artifact or interpreter installation. The runner
recomputed all seven files; their bytes match the frozen expectations. An exact
rebuild and a second clean execution match the saved receipt.

These hooks are an execution check, not a hardened OS sandbox for hostile code.
The supplied code was inspected and is owned by this project. Tests deliberately
attempted a package-external read, socket creation and subprocess invocation;
each was refused. Other tests cover altered evidence, missing files/seeds,
nonfinite values, symlinks, private identifiers, archive paths and overwrite
refusal. Forty-two scoped tests pass, including seventeen prior manuscript
tests. This is not the full legacy suite.

The first test run failed on an incorrectly guessed field-count assertion
(expected more than 1,200 versus the actual 1,062). The exact derived count was
corrected; no scientific value or acceptance tolerance changed. The initial
`-I`-only draft was retained privately; the final execution also disables site
packages. Old manuscripts, source reports and model checkpoints are unchanged.

## What This Does Not Establish

This is **aggregate reporting reproducibility**, not full training replication,
independent verification by another team, a new PyTorch result, or evidence that
the method improved. Archived bootstrap intervals are copied with their scope,
not recalculated. The figures are hash-checked reference files, not regenerated
by the standalone entrypoint. The existing workspace builder can regenerate them.

The archive is a de-identified **draft**, not certified anonymous. Exact numbers,
figures and method descriptions may identify previously public work. Direct
identifier scanning has a fixed marker set; it cannot prove absence of every
possible identity clue. Final human review and the venue's complete current
policies remain required. The archive is not uploaded or submitted. Builder,
tests and light verification records can be public without representing the
archive as a submission-ready anonymous supplement.

Both original primary superiority gates remain failed. All four physical sites
are development-exposed. Missing-label easy protection and independent risk
calibration remain unresolved. Stage5C and SMC are off.

## Reproduction

```bash
.venv-pytorch/bin/python scripts/build_m3w_blinded_reproduction.py --verify
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_blinded_reproduction.py tests/test_m3w_evidence_manuscript.py tests/test_m3w_evidence_manuscript_v2.py
```

To build on a clean checkout with the pinned public reports, omit `--verify`.
The builder refuses to overwrite an existing archive or receipt. Inside the
extracted artifact, run `python3 -I -S reproduce.py --verify --output reproduced`.
Repeat with a different output directory; do not delete source evidence to pass
the checks. No raw data, external source files or checkpoints are fetched.

## Next Scientific Action

Independent source admission and calibration/confirmation roles remain the
largest gap. The primary-source TRAF follow-up did not resolve local geometry;
see [source follow-up](source_followup.md). Existing DUT source-use/exposure,
DroneCrowd annotation-download permission and independent-role decisions remain
pending. These are not changed by a working reproduction package. Do not use
another documentation revision or exposed-site threshold sweep as a substitute.

Local/GitHub starting state was `2c52e119`. Local SSH configuration still does not
establish a CREATE project connection; current remote assets/jobs are unverified,
not absent. No remote job was submitted. This subtask needed no GPU or HPC, and
all its processes completed. The research goal remains active and unmet.
