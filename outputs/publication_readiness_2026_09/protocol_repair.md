# M3W Evidence Protocol Repair

Date: 2026-09-16. Result source: fresh_run code/content audit and regression tests.
No neural training, new performance estimate, or deployment promotion is claimed.

## What changed the conclusion

The initial publication audit checked file identities and found test-dependent selection.
The next audit compared raw file bytes, parsed numeric trajectory rows, and cached window geometry.
It found real split contamination, not merely a hypothetical risk.

| Scope | Finding | Consequence |
| --- | --- | --- |
| Stage35/37 original split | TrajNet students001 in val and UCY students001 in test are byte-identical; 47,223 cached windows have identical geometry | Val-selected thresholds and test results do not have independent recordings |
| Stage35/37 original test | Two paths for zara03 contribute 9,540 identical windows each | Duplicate weighting and inflated apparent evaluation sample size |
| Stage43/44 new split | 47,223 train/val duplicates and 9,540 train/test duplicates | Path-level disjointness is not recording-level independence |
| Stage43/44 inherited teacher | 17,070 new val rows and 78,270 new test rows came from the old teacher train | Cached supervised teacher outputs must be refit under the new split |
| Stage37 selector | Final variant ranking uses test_metrics | Bootstrap on the selected result does not remove selection bias |
| Legacy t25 | Zero cached t25 rows have endpoint frame delta exactly 25 | Requested horizons and actual target sampling must be separated |

See [current split content audit](recording_lineage_audit.md) and
[original split content audit](stage35_recording_lineage_audit.md).
Numeric comparison uses frame/agent/x/y identity at five decimal places, without
coordinate transformations. Byte-equality findings do not depend on that rounding.
Transformed or reindexed copies may remain, so absence of additional matches is not
a complete cross-format deduplication certificate.

## Implemented repairs

1. WorldCore architecture selection reads validation policy metrics only. Missing
   validation evidence fails closed; unsafe or nonfinite candidates are excluded.
2. The repair trigger reads validation only. The training function no longer accepts
   a test split, predicts on test, or returns test metrics.
3. Development data are loaded separately. The selected variant, policy parameters
   and checkpoint hashes are written to a selection lock before test prediction
   data are loaded. Evaluation verifies checkpoint hashes before use.
4. Policy search includes the unchanged floor at objective zero. A negative-gain
   candidate no longer wins just because every switching candidate is bad.
5. A fresh content/teacher-lineage preflight runs before training or writing old
   artifacts. The current legacy caches fail it and are blocked, as intended.
6. New checkpoints include feature normalizers and model dimensions. The arm64
   runtime guard executes before this module imports torch.
7. Input-exclusion declarations are no longer named a complete no-leakage gate.
   Recording/teacher evidence and validation-only selection are separate checks;
   passing implementation checks never sets submission_ready or confirmatory_evidence.

The historical Stage44 implementation can still be audited from commit `0376ed6b`.
The evidence-audit script pins that historical source, so later repairs do not
silently change which code its diagnosis describes. Historical results/checkpoints
were not overwritten. The repaired pipeline still labels the historical test as exposed.

## Verification

Regression tests first reproduced five failures in the old selection implementation:
test-dependent ranking, inability to select without test, missing-val fallback to
test, selection of an unsafe validation candidate, and negative-gain switching.
They passed after the targeted fixes. Tests also check numeric alias detection,
teacher exposure, missing-input rejection, preflight-before-output ordering, and
selection-lock-before-test-loading ordering.

Focused regression result: **23 passed in 14.61 seconds** across the four test
files below. This is not a full-repository test run or a real-data prediction test.
The pinned historical source audit also replayed successfully in a separate
temporary output directory, retaining all seven checkpoint and twelve input identities.

```bash
.venv-pytorch/bin/python scripts/audit_m3w_recording_lineage.py
.venv-pytorch/bin/python scripts/audit_m3w_recording_lineage.py --cache-layout stage35
.venv-pytorch/bin/python -m pytest tests/test_m3w_recording_lineage.py tests/test_stage44_worldcore.py tests/test_stage43_full_waypoint_latent_dynamics.py tests/test_stage43_full_waypoint_supervision_cache.py -q
.venv-pytorch/bin/python scripts/audit_m3w_submission_evidence.py --output-dir /tmp/m3w_historical_evidence_replay
```

Negative integration check: `run_stage44_worldcore.py --quick` exits with
`M3W lineage preflight failed` before training/output mutation. This is an expected
rejection of contaminated caches, not a runtime hang or a completed training run.

## Still required

- Declare original-recording aliases, including coordinate-transformed copies;
  choose one canonical representation and quarantine unresolved identities.
- Define the primary published benchmark protocol, separately from legacy
  raw-frame diagnostics. Do not silently rename approximate horizons.
- Rebuild histories, train-only context, normalizers, labels and teachers within
  each new training fold. Teacher gain/harm supervision needs out-of-fold predictions.
- Repair JEPA target learning and replace proxy-labelled heads only after the
  data boundary is trustworthy. Neither repair nor a new neural run happened here.
- Refit standard baselines and predictors, freeze validation choices, and assess
  recording/scene-level uncertainty. An exposed historical split is not made
  untouched by repartitioning it; independent confirmation is still missing.

This is evidence for correcting the research process, not a new paper method or
performance success. Historical external claims require revalidation. No claim is
made that every SDD result was audited or invalidated. Dataset-local/raw-frame
only; Stage5C and SMC remain disabled.
