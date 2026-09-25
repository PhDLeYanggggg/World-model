# Execution and Reproducibility

Registration commit `8f2aca435820cbc79cae21cdc2da809c73bd06ea` was pushed before decision construction.
Native arm64 CPU4/inter-op1/workers0. No new neural training or support-threshold refit.

```json
{
  "checks": {
    "saved_decisions_verified": 936,
    "old_decision_arrays_exact": 288,
    "independent_coordinate_arrays": 144,
    "independent_metric_reductions": 12528,
    "old_metrics_exact": 1728,
    "partition_equalities": 4608
  },
  "barrier": {
    "decisions_completed": {
      "batch": "2026-09-25T14:54:57Z",
      "fitting": "2026-09-25T14:59:24Z"
    },
    "first_new_readout": "2026-09-25T15:02:32Z"
  }
}
```

Both full decision banks froze before the first new outcome readout. Scalar source-set and query-ranking replay
checks every saved policy. Old stop/joint/risk/random decisions and metrics replay exactly.
Source/producer exclusions and train-only support boxes retain the prior audited boundaries.
Private decisions and result receipts are immutable and bound to config/code/schema hashes.

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_support_factorization.py --mode batch --phase decide
.venv-pytorch/bin/python scripts/run_m3w_european_support_factorization.py --mode fitting --phase decide
.venv-pytorch/bin/python scripts/run_m3w_european_support_factorization.py --mode batch --phase evaluate --resume
.venv-pytorch/bin/python scripts/run_m3w_european_support_factorization.py --mode fitting --phase evaluate --resume
.venv-pytorch/bin/python scripts/report_m3w_european_support_factorization.py
```

Authorized private source data and score banks are required. Aggregate-only Git cannot recreate raw predictions.
The scoped test manifest is explicit; no claim that the unrelated legacy suite ran.
Opened sources remain development sources. Independent model selection, calibration and confirmation remain closed.
No human-gold, physical-safety, true-3D or foundation claim. Stage5C/SMC remain off.
