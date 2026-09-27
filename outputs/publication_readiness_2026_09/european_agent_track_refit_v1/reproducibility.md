# Reproduction and Recovery

This is a replay from hash-bound admitted source assets, not a cold rebuild
from newly downloaded data or an anonymous-submission archive. No raw data,
private checkpoints or prediction arrays are in Git. Parent seals bind the
source code, schema, split, lineage and private artifact identities.

## Initial Execution

Use the native arm64 `.venv-pytorch/bin/python`. Architecture rejection occurs
before Torch import. Training uses CPU4/interop1, zero workers. The real pilot
is part of the eventual first endpoint, not an import-only runtime check.

```bash
PRIVATE=data/stage_cvpr2027_experiments/european_agent_track_refit_v1
run_logged() {
    local phase="$1"
    shift
    if test -e "$PRIVATE/$phase.log"; then
        printf 'Preserve bound logs; use a new audit log.\n'
        return 1
    fi
    /usr/bin/time -l .venv-pytorch/bin/python \
        scripts/run_m3w_european_agent_track_refit.py --phase "$phase" "$@" \
        > "$PRIVATE/$phase.log" 2>&1
}
.venv-pytorch/bin/python scripts/run_m3w_european_agent_track_refit.py --phase register
# Commit the registration and bound design before fitting.
run_logged pilot
run_logged train --resume
run_logged predict
# Commit prediction_freeze.json before comparative scoring.
run_logged evaluate
run_logged replay
run_logged verify_eval
.venv-pytorch/bin/python scripts/probe_m3w_trained_agent_track.py
.venv-pytorch/bin/python scripts/diagnose_m3w_agent_track_envelope.py
.venv-pytorch/bin/python scripts/report_m3w_european_agent_track_refit.py
.venv-pytorch/bin/python scripts/report_m3w_agent_track_operations.py
.venv-pytorch/bin/python scripts/verify_m3w_european_agent_track_refit.py
```

The final seal is write-once. Validate hashes rather than replacing it. After
sealing, direct fresh replay output to a new audit log, leaving original timing
receipts untouched. Never delete a checkpoint just to force a rerun.

## Interrupted Training

Inspect the PID and live process, log and heartbeat first. Observation timeouts
alone do not mean terminal failure. The runner's file lock prevents duplicate
local workers. With no live worker, resume the same training command using
`--resume` and a new log name. Endpoint identity, sampler, settings and cached
controls must match. Atomic checkpoints every200 updates retain optimizer and
RNG states. Each50 updates reports progress; the process preserves10GiB disk.

Existing endpoints are verified, not retrained. A changed scientific design
requires a new versioned experiment, not mutation of this registration.
If the model no longer fits local resources, preserve checkpoints and consult
the approved CREATE handoff, inspect jobs/resources read-only, then submit via
the scheduler with a recorded job ID. No HPC work was launched by these commands.

## Interpretation

Code tests establish causality, identity and recovery behavior, not a research
gain. A scoring replay establishes repeatability, not independent confirmation.
Main uncertainty is locality-resampled, conditional on exposed source development;
the two producer contexts and three seeds within a locality are dependent.
No independent outcome, risk-policy deployment, Stage5C or SMC is authorized
by a passing verification receipt.
