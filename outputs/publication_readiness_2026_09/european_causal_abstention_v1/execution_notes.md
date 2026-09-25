# Execution and Reproducibility

Registration commit: `3ff17937`, pushed before decision construction.
Native arm64 Python; CPU four threads, inter-op one, no loader multiprocessing.
No new neural weights or trajectory forecasters fitted; support boxes and decisions were freshly built.

```json
{
  "checks": {
    "saved_decisions_verified": 720,
    "independent_coordinate_arrays": 144,
    "independent_metric_reductions": 8208,
    "original_metrics_exact": 432
  },
  "barrier": {
    "decisions_completed": {
      "batch": "2026-09-25T14:15:48Z",
      "fitting": "2026-09-25T14:16:53Z"
    },
    "first_new_readout": "2026-09-25T14:18:39Z"
  }
}
```

The verifier hash-checks the prior 234-head experiment, both new decision banks, source exclusion,
and all 36 complete result receipts. Scalar box/query ranking replay verifies all 720 saved policies.
Only fitting rows contribute support boxes. No held labels determine guard decisions or frame counts.
All eight-locality rosters retained. Risk-ranked and random controls match each guard within recording/current frame.

Private artifacts include immutable per-group decisions, support metadata, results, identity, lock, heartbeat and events.
Re-running evaluation requires `--resume`. The full legacy test suite is not claimed here.

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_causal_abstention.py --mode batch --phase decide
.venv-pytorch/bin/python scripts/run_m3w_european_causal_abstention.py --mode fitting --phase decide
.venv-pytorch/bin/python scripts/run_m3w_european_causal_abstention.py --mode batch --phase evaluate --resume
.venv-pytorch/bin/python scripts/run_m3w_european_causal_abstention.py --mode fitting --phase evaluate --resume
.venv-pytorch/bin/python scripts/report_m3w_european_causal_abstention.py
```

Existing authorized source data and private score banks are required; aggregate-only Git contents cannot recreate them.
Independent model selection, risk calibration and confirmation remain closed. Stage5C/SMC remain off.
Image-pixel obs8/pred12 rawstride12; not historical raw-t50, metric/seconds, human gold, true 3D or foundation evidence.
