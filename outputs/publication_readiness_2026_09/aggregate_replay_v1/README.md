# Standalone Aggregate Evidence Replay

The evidence manuscript now has a small replay package that runs without a
repository checkout, external datasets, model weights or cluster access. It
contains eight pinned public aggregate/protocol inputs and a standard-library
Python program. The program rebuilds three tables, eleven EuropeanSquares
contrasts and four separate SDD contrasts exactly.

## Verified Scope

Verification status: **VERIFIED for aggregate replay only**. Construction and
isolated execution are `fresh_run`; the underlying scientific numbers remain
`cached_verified`. Both verification-only and output-writing modes ran, and the
three output files match the manuscript exports byte for byte. Thirty-five
focused tests passed, including corruption rejection, package allowlisting,
identifier-scan sensitivity and the existing manuscript consistency tests.

The 51,282-byte archive is local and intentionally excluded from Git. Its hash
and every member are recorded in [verification.json](verification.json).
Rebuild and check it from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m scripts.build_m3w_aggregate_replay_package
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m scripts.build_m3w_aggregate_replay_package --check
```

The archive is written to
`data/stage_cvpr2027_experiments/aggregate_replay_v1/review_bundle.zip`.
After extraction, `python reproduce.py --check` verifies the package and exact
outputs; `python reproduce.py` additionally writes the replayed files. Neither
command contacts a network service or imports a scientific-computing library.

## Evidence Boundaries

This is not model-training reproduction, raw-row verification or an independent
replication. Confidence intervals are taken from the pinned studies, not
bootstrapped again. Prior exposure, failed primary comparisons, seven upper-bound
easy-risk violations in the extended policy and the selected temporal contrast's
eleven-locality support remain visible. Nothing is promoted to a successful or
independently calibrated policy.

The package preserves the manuscript's historical snapshot, not current job
status. Current training progress remains in the main results ledger. Its
known-identifier scan removes no scientific content and detects listed personal
paths, account strings, emails and repository URLs. Source text and hashes remain
linkable to public material, so venue-compliant anonymity is **not certified**.
The full experimental and anonymous submission packages are still incomplete.

No new training, forecast evaluation, bootstrap, independent-role access,
deployment, Stage5C or SMC execution occurred. Coordinate/time/label claims
remain pixel/raw-frame/detector-or-inferred, not metric, seconds or human gold.
