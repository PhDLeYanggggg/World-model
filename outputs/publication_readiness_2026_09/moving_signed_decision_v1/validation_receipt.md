# Validation Receipt

- Registration 8f3ef2b5 precedes all new training.
- Pilot e5aa5130: both real loss arms resume exactly, 400 updates including repeats.
- Training freeze 0cd9dd30 precedes held readout; six primary fits complete
  12,000 updates plus a 2,000-update exact treatment refit.
- Training PID 84628: observed exit 0, 517.022363875 seconds, peak RSS
  7,521,599,488 bytes. Each parameter completes 2,000 AdamW updates, common
  encoder weights change, weights remain finite. All three fresh MSE control
  models exactly reproduce their old model weights and sampling.
- Evaluation PID 85449: observed exit 0, 46.572871250 seconds.
- Verification PID 85536: observed exit 0, 45.659445958 seconds.
- All three new prediction/action/metric blocks match exactly. Old cached
  action hashes match and recomputed old metrics match exactly.
- Analysis SHA256:
  `2d9d014f07ba2be9910e463e045f2ab6a1a22453f7aa6c09a621d608d89d681c`.
- Both fresh report runs observed exit 0. All 30 existing public evidence
  files are byte-identical before and after the second report run. This receipt
  is added afterward, not included in that 30-file comparison.
- README/state metrics were checked against saved analysis and verification.
- Focused validation: 26 tests pass, covering loss-sign consistency, unchanged
  unknown-bound MSE, empty queries, actual parameter updates, exact resume,
  source exclusion, causal feature provenance and NA/reporting boundaries.
- Full historical test suite not rerun. This receipt is engineering evidence,
  not a scientific safety pass or independent statistical confirmation.

```sh
.venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_signed_decision_reporting.py \
  tests/test_m3w_signed_decision.py \
  tests/test_m3w_source_error_priority.py \
  tests/test_m3w_admission_priority.py \
  tests/test_m3w_kinematic_risk.py \
  tests/test_m3w_kinematic_feature_provenance.py \
  tests/test_m3w_direct_tail_risk.py
```

No evaluation-rule changes, identity certification, reserved H readout,
deployment upgrade, Stage5C execution or SMC. Checkpoints/caches remain local.
The pre-existing unrelated staged index was preserved during scoped commits.
