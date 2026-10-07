# Revision-Four Aggregate Replay

This package reproduces the six tables and18 temporal contrast/metric rows in
the integrated development manuscript. It retains the failed neural-control
comparisons, risk violations, unknown outcomes and undefined selected-cohort
comparisons. SDD, European development and TRAIN resubstitution are not pooled.

## Verified Scope

- Fourteen included source files have fixed SHA256 hashes.
- The exporter checks216 linked training receipts and72 readout receipts in the
  repository. The standalone archive does not include or reverify these receipts.
- An isolated Python process, with no repository imports or numerical packages,
  reproduces `evidence.json`, `tables.md` and `temporal_contrasts.csv` exactly.
- Thirty-eight scoped package/manuscript tests pass. The full legacy suite was
  not rerun because no training or evaluation implementation changed.
- The85,571-byte archive has SHA256
  `f80eaf755f0abeefea5b5fe43d6a3068d9a93a4c468c5ced0a1de5298e886c36`.

## Reproduction

From the project root, regenerate or verify the local bundle:

```bash
.venv-pytorch/bin/python -m scripts.build_m3w_aggregate_replay_v2
.venv-pytorch/bin/python -m scripts.build_m3w_aggregate_replay_v2 --check
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_aggregate_replay_v2.py
```

The archive is written to
`data/stage_cvpr2027_experiments/aggregate_replay_v2/review_bundle.zip`.
After extraction, use Python3.10 or later:

```bash
python reproduce.py --check
python reproduce.py
```

The first command checks the included hashes and aggregate outputs. The second
also writes byte-identical outputs into `replayed/`. Neither needs network,
CREATE, a repository checkout, raw trajectories or checkpoints.

## Limits

Export and isolated replay are `fresh_run`; scientific numbers are
`cached_verified`. No model training, forecast evaluation, bootstrap or new
independent test was run. Source confidence intervals are retained, not
re-estimated. Public receipt integrity is not checkpoint or raw-label validation.
The older nested `prior_evidence` is a historical snapshot; top-level revision4
records describe the completed temporal experiment.

The package excludes known account identifiers and home paths by mechanical
scan, but public-source hashes remain linkable. It is identity-minimized
preparation, not certified anonymous submission material or full experimental
reproduction. The easy-harm-deviance successor still has no completed scientific
readout. No deployment promotion follows.

Coordinates remain pixel/image-local and horizons raw-frame; normalized scores
are identified separately. Labels are not human gold. No metric, seconds,
true-3D or foundation claim is made. Stage5C and SMC stay off.
