# Reproducing the Matched Loss Control

## Registration

Pre-fit commit `ee3e8667`, config `configs/m3w_source_motion_candidate_v1.json`,
SHA256 `e40a9c690a7e00e303c2ac27af36ca06fb6133bf2be24974c4bf381ce250963d`.
Ten direct frozen bindings and inherited control/data bindings remain unchanged.
Direction null registered separately in `4e21587d`, during training and before
its computation; it is explicitly post-hoc.

## Commands

Run from the repository root in the native arm64 environment:

```bash
.venv-pytorch/bin/python scripts/run_m3w_source_motion_candidate.py --registration configs/m3w_source_motion_candidate_v1.json --audit-only
.venv-pytorch/bin/python scripts/run_m3w_source_motion_candidate.py --registration configs/m3w_source_motion_candidate_v1.json
.venv-pytorch/bin/python scripts/run_m3w_source_motion_candidate.py --registration configs/m3w_source_motion_candidate_v1.json --replay
.venv-pytorch/bin/python scripts/analyze_m3w_source_motion_candidate.py --registration configs/m3w_source_motion_candidate_v1.json
.venv-pytorch/bin/python scripts/verify_m3w_source_motion_candidate.py --registration configs/m3w_source_motion_candidate_v1.json
.venv-pytorch/bin/python scripts/diagnose_m3w_motion_direction.py --registration configs/m3w_source_motion_candidate_v1.json
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_source_motion_candidate.py tests/test_m3w_source_crossfit.py tests/test_m3w_source_modality_continuation.py tests/test_m3w_source_cost_dynamics.py tests/test_m3w_recording_diagnostic.py tests/test_m3w_crossfit_cost_diagnostic.py tests/test_m3w_direction_null.py
```

The optional included pilot is `--trial coupa_seed17 --stop-at 100`; it does
not score held queries. A normal invocation resumes the same checkpoint with
optimizer, draw counts, sampler and Torch RNG state. Do not run duplicate copies
or remove completed receipts. A completed invocation performs zero new updates.
Changing a frozen dependency must fail, not silently reinterpret an experiment.

Replay reforecasts each training complement and held-site cohort and demands
exact equality. The independent verifier checks roles, normalization, costs,
finite bounds, all matched sampler states, OOF coverage and immutable resume.
Target poisoning concerns loaded loss labels, not raw annotation acquisition.

## Completed Receipt

Torch 2.12.0, NumPy 2.4.6, four CPU threads, one inter-op thread, zero workers,
native arm64. Checkpoints and heartbeat every 200 updates. Pilot PID 59254
completed 100 updates in 4.923036 seconds. Main PID 59275 exited zero with
119,900 additional updates; total 120,000. Main log span 5,662.259590 seconds
(94.371 minutes), summed fit 5,604.923353 seconds including pilot. Approximate
observed RSS 2.6 GiB, not a continuous peak-memory measurement.

Replay PID 67029 exited zero, all 12 models exact. Fixed analysis, verifier and
direction null exited zero. Resume child PID 67211 adds zero updates and preserves 68
artifacts. Ninety-six loaded-label poisoning checks pass, representing eight
queries per model rather than 96 unique queries. Figure inspected. 34 focused tests
pass; full legacy suite not rerun. Matplotlib built its font cache successfully.
All required processes are terminal. No CREATE job submitted; current remote
assets and scheduler state remain unverified.

| Artifact | SHA256 |
| --- | --- |
| report.json | f9c25953b632c5022fc58ba0de9fbdabb7cbf5440fa0f26f6c37bc016bca381f |
| analysis.json | 08ff58f211849f7eea416ae9b5a0dc87d7a5ee21d3aa131fd69d9542dd336697 |
| replay.json | 3287c2ccbcbeba9b5b0e92ede215ea4f3098aa084984b16255d4758741f5b0d8 |
| verification.json | 7010afc9c4114673ba8386956f86afef4346558179ee51aefb927734058ba4dc |

## Assets and Interpretation

Private checkpoints, per-query predictions, OOF costs and logs remain under
`data/stage_cvpr2027_experiments/source_motion_candidate_v1/`. Public code/config/
aggregate reports do not include these files or third-party data. Reproduction
requires the matching locally acquired data and previous verified control
artifacts; a public clone alone is not self-contained reproduction.

Use the original equal-site normalized ADE, not oracle selection or favorable
subset scores, for the primary contrast. Bootstrap draws 2,000 / seed 38113 describe
four explored sites with shared training folds; they do not certify independent
generalization. See [conclusions](conclusions.md) for all negative slices and
[gates](gates.md) for evidence that remains missing. No new deployment,
metric/seconds claim, Stage5C execution or SMC.
