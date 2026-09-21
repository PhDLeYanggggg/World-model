# Execution and Replay

## Material Passport

Status: completed real-data decisions and evaluation; verified arithmetic and
replay. Research verdict: no non-additive mechanism lift; not publication-ready.
Scope: source-only development, frozen native forecasters/cost heads, no training.
Runtime: native arm64 `.venv-pytorch`, CPU4/inter-op1/workers0. No CREATE job was
submitted or newly checked for this cached-forecast comparison.

## Fixed Sequence

Registration `f4d00832` was pushed before a 256-scene-query pilot. The remaining
62,540 scene/seed queries resumed from its checkpoints. Completion retains all
62,796 queries and 291 chunk receipts. All decisions preceded future target-array
readout. Files under `data/stage_cvpr2027_experiments/native_joint_controls_v1/`
are local only; do not commit them.

```sh
.venv-pytorch/bin/python scripts/run_m3w_native_joint_controls.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_native_joint_controls.py --resume
.venv-pytorch/bin/python scripts/run_m3w_native_joint_controls.py --evaluate
.venv-pytorch/bin/python scripts/run_m3w_native_joint_controls.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_native_joint_controls.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_native_*.py tests/test_m3w_interaction_controls.py tests/test_m3w_joint_intervention.py
```

Existing immutable chunks are verified on resume, not overwritten. `--verify`
recomputes decisions without future arrays and compares every saved choice.
Identity binds 610 sources; code/config/registration edits must use a new run
identity rather than silently resuming this experiment. A file lock prevents
concurrent decision writers. The local heartbeat records PID and group progress;
historical PID presence is not evidence of a currently running job.

## Observed Runs

- Decision pilot PID 82379, full-resume PID 82468: normal exits, all groups complete.
- Outcome readout PID 83030: normal exit; fresh `analysis.json`.
- Full decision replay PID 83102: normal exit; 5,272,680 choices agree.
- Separate verifier: all 920 scene reductions and all 88 real-query optima agree.
- Targeted regression suite: 175 tests passed in 4.19 seconds.
- The first verifier attempt stopped before writing a success record because it
  subtracted float32 cost columns before widening them. The frozen policy widens
  first. Matching that arithmetic repaired the verifier; neither policy bytes
  nor numerical/risk tolerances changed. No experimental result was replaced.

The resumed decisions and the complete replay each took approximately 50 seconds
of cached inference/control work. These are not neural fitting or end-to-end
training timings. No long live job was stopped for slowness. Full historical
repository tests were not rerun; the targeted suite covers the changed path.

## Interpretation Checks

The CI uses physical scenes, not independent-window assumptions. Four explored
scenes and three seeds do not constitute independent risk calibration. A [0, 0]
paired CI follows identical selections, not proof of equivalent population
performance. No cherry-picked seed, test threshold, causal claim, physical
collision claim, prospective safety bound or equivalence claim is made. Counts
repeat across seeds and overlapping windows; unknown outcomes remain unknown.
Full-count and half-scene-uniform structural nulls are disclosed. Neither a
solver certificate nor passing tests means that the research hypothesis passed.
