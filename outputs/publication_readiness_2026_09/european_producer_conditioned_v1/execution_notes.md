# Training And Reproducibility

Protocol commit `3f5f134a1584bf0f119b9d2c83ff4b8d450e9284` was pushed before the pilot and full fit.
Native arm64 CPU4/inter-op1/workers0. Each head has 382 inputs and width64. Utility has 24,642 parameters; risk has 24,707.
The actual 100-step utility/risk pilot resumed inside the 2,000-step budget. Identical training draws, labels, envelopes, first380 preprocessors and fixed ranking denominators were checked across arms.
All unknown-label supervised draw counts are zero. Both 4,096-row producer prefixes replay exactly from every new checkpoint (216 prefix checks).
Sampled losses use changing batches. Only risk has a fixed training-batch trace; neither trace is validation or test success.

| Arm / head | Heads | Updates | Sampled first loss | Sampled final loss | Fixed training-batch decrease |
|---|---:|---:|---:|---:|---:|
| global / utility | 18 | 36000 | 0.0857 to 0.4251 | 0.0379 to 0.3507 | not measured |
| global / risk | 18 | 36000 | 0.2538 to 3.5037 | 0.2022 to 2.7242 | 16 |
| producer / utility | 18 | 36000 | 0.0857 to 0.4251 | 0.0358 to 0.3525 | not measured |
| producer / risk | 18 | 36000 | 0.2538 to 3.5037 | 0.2040 to 2.6925 | 16 |
| placebo / utility | 18 | 36000 | 0.0857 to 0.4251 | 0.0381 to 0.3443 | not measured |
| placebo / risk | 18 | 36000 | 0.2538 to 3.5037 | 0.2014 to 2.7186 | 16 |

```json
{
  "checks": {
    "saved_decisions_verified": 180,
    "coordinate_arrays_verified": 216,
    "metric_reductions_verified": 3402,
    "old_anchor_metrics_exact": 90
  },
  "readout_barrier": {
    "banks": "2026-09-25T15:50:02Z",
    "pilot": "2026-09-25T15:50:35Z",
    "train": "2026-09-25T16:00:02Z",
    "decide": "2026-09-25T16:02:43Z",
    "first_new_readout": "2026-09-25T16:02:54Z"
  }
}
```

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_producer_conditioned.py --phase prepare
.venv-pytorch/bin/python scripts/run_m3w_european_producer_conditioned.py --phase banks
.venv-pytorch/bin/python scripts/run_m3w_european_producer_conditioned.py --phase pilot
.venv-pytorch/bin/python scripts/run_m3w_european_producer_conditioned.py --phase train --resume
.venv-pytorch/bin/python scripts/run_m3w_european_producer_conditioned.py --phase decide
.venv-pytorch/bin/python scripts/run_m3w_european_producer_conditioned.py --phase evaluate --resume
.venv-pytorch/bin/python scripts/report_m3w_european_producer_conditioned.py
```

For an existing completed directory, run the reporter for verification. Do not restart a pilot over a completed training checkpoint.
Private source files, packed arrays and parent checkpoint banks are required to recreate predictions. Public aggregates alone do not reproduce private forecasts.
A fixed final checkpoint was preregistered; no post-readout checkpoint/threshold selection. No reserved role was opened.
