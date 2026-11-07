# CV-only and dual-reference forecast training

I trained the same 3,560-parameter causal forecast head under two losses to test whether accounting for OLS4 as well as CV produces a better path at the same intervention budget. There were six fresh Torch fits, with 900 updates per fit, matched initial weights and draws at seeds 17, 29 and 43, and 69,746 fitting rows from nine source groups disjoint from development. Each forecast was frozen before scoring the 125,540 exposed-development windows across 34 recordings and 14 source groups. The primary comparison replaced the paths, not the recipients, on all 9,675 original gain-policy interventions across the three seeds (2.569%). The comparison and source-group bootstrap replayed exactly.

| Comparison | Gain on observed labels (pp) | Conservative completion lower (pp) | 95% source-bootstrap CI for lower (pp) |
| --- | ---: | ---: | ---: |
| Dual vs CV-only, matched recipients (primary) | +0.068 | -0.287 | [-0.816, +0.060] |
| Dual vs original gain policy, matched recipients | -0.234 | -0.785 | [-1.714, -0.048] |
| CV-only vs original gain policy, matched recipients | -0.302 | -1.057 | [-2.179, -0.113] |
| Dual vs CV-only, full roster | +0.998 | -3.231 | [-7.628, -0.314] |
| Dual vs OLS4, full roster | +11.867 | +2.271 | [-5.303, +9.542] |

Gains are measured against the known CV ADE mass. The completion lower accounts for wholly unknown rows under the existing debit; the comparison does not recover missing positions on partially observed tracks. The bootstrap averages seeds within source before 3,000 paired source-group draws.

The observed-label advantage does not establish a safer or better forecast. Only 12/102 matched dual views and 13/102 matched CV-only views meet the existing per-view 2% requirements; 23 views per arm have undefined easy risk. Matched dual has 11 views with observed easy net degradation above 2%, compared with 10 for CV-only. Worst observed easy degradation is 11.75% for dual and 9.44% for CV-only. Neither arm is a deployment candidate.

This specific loss changes the gradient through the unknown rows: on known rows, subtracting CV or OLS4 ADE only changes the reference constant, so both terms have the same prediction gradient there. On the matched recipients the dual arm improves the observed contrast by 0.068 points, but the added disagreement debit of 0.355 points more than offsets it. This tests a particular way to charge ambiguity, not whether OLS4 can help as a feature, a control, or an explicit anchor on a separately trained regime.

Result origin: `fresh_run` (training, prediction, scoring and exact replay); `cached_verified` (prior roster, causal comparison bank, recipient policy and protocol bindings). DroneCrowd confirmation was not run or opened. No metric or seconds claim, no deployment change, no Stage5C or SMC.

## Reproduction

Run from the repository with the native Torch environment and the previously registered local fitting and development assets. The protocol and registration must be committed before a fresh run; all paths and checkpoints stay local. An interrupted fit resumes from its last atomic checkpoint via `train --resume`.

```bash
.venv-pytorch/bin/python scripts/run_m3w_dual_reference_forecast.py register
.venv-pytorch/bin/python scripts/run_m3w_dual_reference_forecast.py train
.venv-pytorch/bin/python scripts/run_m3w_dual_reference_forecast.py infer
.venv-pytorch/bin/python scripts/run_m3w_dual_reference_forecast.py score
.venv-pytorch/bin/python scripts/run_m3w_dual_reference_forecast.py replay
```

[Fixed protocol](protocol.md) | [Aggregate metrics](analysis.json) | [Replay receipt](replay_execution.json)
