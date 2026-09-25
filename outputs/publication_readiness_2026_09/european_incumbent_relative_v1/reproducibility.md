# Reproduction And Data Boundaries

Registration2ebde873 was pushed before real fitting. Native arm64 CPU4/inter-op1/workers0.
Pilot100 steps resumed inside the2,000-step budget;144 checkpoint4096-row prefix replays and72 complete ridge score replays.
Cached forecasters retain prior verified lineage;144 incumbent-head inference calls and72 original-policy replays on B/C passed this round.
Changing-batch loss is not validation. Fixed risk traces are training diagnostics, not validation evidence.
No C fitting statistics; A easy/hard cutoffs, B supervised losses/normalization, C opened development readout only.

```json
{
  "checks": {
    "saved_decisions_verified": 288,
    "coordinate_arrays_verified": 144,
    "metric_reductions_verified": 3600
  },
  "barrier": {
    "prepare": "2026-09-25T17:18:45Z",
    "pilot": "2026-09-25T17:21:22Z",
    "train": "2026-09-25T17:33:25Z",
    "decide": "2026-09-25T17:34:56Z",
    "first_new_readout": "2026-09-25T17:36:02Z"
  }
}
```

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_incumbent_relative.py --phase prepare
.venv-pytorch/bin/python scripts/run_m3w_european_incumbent_relative.py --phase pilot
.venv-pytorch/bin/python scripts/run_m3w_european_incumbent_relative.py --phase train --resume
.venv-pytorch/bin/python scripts/run_m3w_european_incumbent_relative.py --phase decide
.venv-pytorch/bin/python scripts/run_m3w_european_incumbent_relative.py --phase evaluate --resume
.venv-pytorch/bin/python scripts/report_m3w_european_incumbent_relative.py
```

For a completed run use the reporter. Resume only the identical interrupted run; do not restart the pilot over completed heads.
Public code/config/aggregate receipts do not contain private raw tracks, forecast caches or checkpoints. Full training requires the verified local asset chain.
The exact scoped test list and hashes are recorded in completion_checks.json. Full legacy suite is not_run.
