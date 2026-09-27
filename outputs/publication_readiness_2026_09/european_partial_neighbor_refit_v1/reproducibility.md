# Reproduction and Recovery

Use the repository's native arm64 `.venv-pytorch/bin/python`. The runner sets
four Torch compute threads, one interop thread and no DataLoader workers.
It rejects Rosetta before importing Torch. One local file lock prevents two
processes from mutating the same experiment. No CREATE job is required for
this bounded local fit; the queue check was read-only.

## Fixed Execution

```bash
PRIVATE=data/stage_cvpr2027_experiments/european_partial_neighbor_refit_v1
run_logged() {
    local phase="$1"
    shift
    if test -e "$PRIVATE/$phase.log"; then
        printf 'Preserve the existing bound log; use a separate audit log.\n'
        return 1
    fi
    /usr/bin/time -l .venv-pytorch/bin/python \
        scripts/run_m3w_european_partial_neighbor_refit.py --phase "$phase" "$@" \
        > "$PRIVATE/$phase.log" 2>&1
}
.venv-pytorch/bin/python scripts/run_m3w_european_partial_neighbor_refit.py --phase register
# Commit registration, protocol, configuration and bound source before fitting.
.venv-pytorch/bin/python scripts/run_m3w_european_partial_neighbor_refit.py --phase prepare
run_logged pilot
run_logged control --resume
run_logged train --resume
run_logged predict
# Commit prediction_freeze.json before reading comparative source outcomes.
run_logged evaluate
run_logged replay
run_logged verify_eval
.venv-pytorch/bin/python scripts/report_m3w_european_partial_neighbor_refit.py
.venv-pytorch/bin/python scripts/diagnose_m3w_european_partial_neighbor_refit.py
.venv-pytorch/bin/python scripts/probe_m3w_neighbor_association.py
.venv-pytorch/bin/python scripts/report_m3w_partial_refit_operations.py
.venv-pytorch/bin/python scripts/verify_m3w_european_partial_neighbor_refit.py
```

The comments identify explicit freeze boundaries, not permission to stage
unrelated files. The final verification seal is write-once: once present,
validate its hashes instead of replacing it. All large inputs, predictions,
checkpoints and detailed logs stay under the ignored private experiment path.
These commands describe initial execution. For a sealed-run audit, preserve
the bound original logs and redirect fresh replay output to a new audit path.
Do not overwrite original timing receipts or delete caches to force a rerun.

## Recovery

The private directory is
`data/stage_cvpr2027_experiments/european_partial_neighbor_refit_v1`.
Read `heartbeat.json` and `events.jsonl` for PID, trial, step and UTC progress.
The runner atomically saves model, optimizer, random states, row sampling counts
and loss history every200 updates. Resume the same phase with `--resume`;
completed hash-verified trials are skipped. An interrupted new fit resumes
from its last atomic checkpoint, not from a model chosen by held outcomes.

The100-update pilot belongs to the same4000-update control budget. It is not
an additional training trial. The control endpoint must match the cached
legacy parameters and sampler before any partial-neighbor fitting is allowed.
An exact replay failure is a stop condition, not a reason to relax tolerances.

## Evidence Limits

This runner requires the locally authorized, hash-bound European source
arrays and legacy producer checkpoints. They are not distributed in Git.
Fresh raw-data acquisition and cold regeneration of every historical asset
are not established by a successful local replay. The replay verifies this
version and its existing inputs, prediction banks and reports.

Three seeds and3000 locality bootstrap draws are retained. The source
localities were already used for development; independent selection,
calibration and confirmation stay closed. Source-fold exclusions are real,
but they do not turn repeated development into a fresh final test. Engineering
tests, completed training, predictive gain and deployment safety are separate
claims. Stage5C and SMC remain disabled.
