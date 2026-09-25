# Training And Reproduction

Registration8dd1dd2d was pushed before the pilot and full fit. Native arm64 CPU4/inter-op1/workers0;380 inputs and width64.
The100-update utility/risk pilot resumed inside each2,000-update budget.144 checkpoint prefixes replay exactly;72 ridge score arrays replay fully.
All9 cached neural prefixes,72 old-head replays and36 old decisions passed before fitting. The preflight index error is documented, not hidden.
Changing-batch losses are not validation performance. Risk also has a fixed training-batch trace. No readout-selected checkpoint or threshold.

| Arm / head | Heads | Updates | First sampled loss | Last sampled loss | Fixed training batch decrease |
|---|---:|---:|---:|---:|---:|
| producer_matched / utility | 36 | 72000 | 0.0488 to 0.5579 | 0.0494 to 0.2153 | not measured |
| producer_matched / risk | 36 | 72000 | 0.2610 to 3.0696 | 0.1835 to 2.7490 | 30 |
| oof_control / utility | 36 | 72000 | 0.0857 to 0.4251 | 0.0354 to 0.3276 | not measured |
| oof_control / risk | 36 | 72000 | 0.2505 to 3.5037 | 0.2044 to 2.7819 | 33 |

Some OOF utility supervision repeats across producer rotations; these fresh paired refits are not independent replications. training_summary.json reports distinct fitting input/target bindings.

```json
{
  "checks": {
    "saved_decisions_verified": 180,
    "coordinate_arrays_verified": 144,
    "metric_reductions_verified": 2376
  },
  "barrier": {
    "prepare": "2026-09-25T16:35:34Z",
    "pilot": "2026-09-25T16:36:15Z",
    "train": "2026-09-25T16:49:47Z",
    "decide": "2026-09-25T16:51:57Z",
    "first_new_readout": "2026-09-25T16:52:17Z"
  }
}
```

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_producer_roles.py --phase prepare
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_producer_roles.py --phase pilot
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_producer_roles.py --phase train --resume
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_producer_roles.py --phase decide
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_producer_roles.py --phase evaluate --resume
.venv-pytorch/bin/python scripts/report_m3w_european_fixed_producer_roles.py
```

For an existing completed experiment use the reporter, not another pilot. Active fits can resume with the same configuration and bindings.
Public code/config/aggregate evidence is available; reproducing private forecasts requires the local source/cache/checkpoint chain. No raw tracks or checkpoints are committed.
Full legacy test suite is not run; the exact scoped suite and source hashes appear in completion_checks.json.
