# Validation Receipt: 5 November 2026

## Executed Runs

| Step | PID | Seconds | Observed exit |
|---|---:|---:|---:|
| Real-data resume pilot | 77921 | 36.65 | 0 |
| Six full neural fits and first independent full refit | 78012 | 515.49 | 0 |
| Held SOURCE evaluation | 78786 | 75.74 | 0 |
| Independent inference and readout replay | 78944 | 76.10 | 0 |

The six primary fits contain 12,000 actual optimizer updates. A full first
refit adds 2,000; the pilot adds 400 including repeated/resumed fits. Training
peak RSS is 7,505,887,232 bytes. All six weights freeze before held evaluation.
Native arm64 CPU4/interop1/workers0, no multiprocessing or resource probing.

The checkpoint audit verifies actual common-encoder changes, all parameter
optimizer steps, identical paired initial weights, source roles, preprocessing,
priority masks, targets, sampler/draw hashes and the full independent refit.
All three inference/decision/metric blocks and all previous controls reproduce
exactly. This is reproducibility, not independent scientific confirmation.

Analysis SHA256:
`5987658277c2e21f0db3969e9509dc4b10a161a6fde74681791da0339c2abbf8`.

## Focused Tests

```sh
.venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_admission_reporting.py \
  tests/test_m3w_admission_priority.py \
  tests/test_m3w_admission_loss_diagnosis.py \
  tests/test_m3w_kinematic_risk.py \
  tests/test_m3w_kinematic_feature_provenance.py \
  tests/test_m3w_direct_tail_risk.py
```

Observed exit0: **16 passed in 2.64 seconds**. The initial strengthened
gradient test failed on an inherited zero-target fixture (1 failed/13 passed).
It was corrected before real training, with the first registration preserved
and registration_v2 operative. The fix adds a nonzero target and verifies
both nonzero gradient and exact 10x scaling. Numerical training code did not
change in response to held outcomes. The full historical suite was not rerun.

## Report And State Verification

The unchanged reporter was re-executed in a fresh native interpreter via its
absolute script path. It reread real fitting data, checked actual weights and
recomputed all full fitting losses and risk diagnostics. Observed exit0:
**32 pre-existing public artifacts checked, zero changed**. The post-readout
interpretation and this receipt are separate additions after that snapshot.
Research-state utility, hard, switch, contrast and analysis-hash fields match
the saved analysis exactly.

## Outcome

The prespecified expansion condition fails. Complete original risk/support
is 0/17, not a safety pass. Original 24-group primary remains NA. One-seed
exposed SOURCE data, missing-outcome completion bounds rather than sampling
CI, no new independent H. No identity certificate, deployment change, new
metric/seconds/true3D/foundation claim, Stage5C or SMC.

Only source, configs, reports and lightweight metrics enter Git. Data,
caches and checkpoints remain local. Unrelated staged work is untouched.
