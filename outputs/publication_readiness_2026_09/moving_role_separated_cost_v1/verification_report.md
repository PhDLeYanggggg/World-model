# Execution And Verification Record

All rows below are observed local executions, not scheduler intentions or time
projections. Native arm64 `.venv-pytorch/bin/python`, CPU4/inter-op1/workers0;
no DataLoader multiprocessing, MPS probing or new CREATE submission.

| Phase | PID | Exit | Seconds | Peak RSS bytes | Evidence |
|---|---:|---:|---:|---:|---|
| Full-action preparation | 75378 | 0 | 119.402408 | 4419616768 | Own feature and producer target parity |
| Real resume/resource pilot | 75912 | 0 | 45.154018 | 3651764224 | 128+128 versus continuous256 exact |
| 48 primary fits | 76038 | 0 | 680.135672 | 6164381696 | 196608 updates |
| 48 complete exact refits | 77577 | 0 | 668.216296 | 6157107200 | Model, optimizer, traces and draws exact |
| First allocation attempt | 78626 | 1 | NA | NA | Missing allocation output directory; before first checkpoint/metrics |
| Recovered complete allocation | 78964 | 0 | 155.133950 | 2604351488 | All18 context/arm/seed choices frozen |
| Full original-contract readout | 79399 | 0 | 5.103152 | 1923088384 | All rows and required views retained |
| Actual fixed-result replay | 80653 | 0 | 13.781801 | 2843049984 | Predictions, deterministic choices and metrics exact |

Execution JSON files under `executions/` retain preparation, pilot, training,
verification, failed allocation and recovery. The replay receipt is private:
`data/stage_cvpr2027_experiments/moving_role_separated_cost_v1/replays/replay_1793933927039552000_80653.json`.
It is a scalar record, not a new fit or independent outcome evaluation.

## Definition And Recovery Amendments

1. Before fitting or the actual training pilot, v1 was corrected to make the
   reference branch action-invariant. Both arms changed identically; no outcome
   or fitted weight was opened. v1 remains, training identity binds immutable
   registration_v2.json. Model-independent preparation was hash-verified, not
   relabelled fresh. reference_invariance_amendment.md records the correction.

2. After primary fitting, one broader synthetic test failed and22 passed because
   an unseeded action-invariance fixture demanded bitwise equality despite
   one-ulp float32 batch effects (maximum5.960464477539063e-8). Verification-only
   v3 fixes the fixture seed, uses eight float32 epsilons of relative forward
   tolerance with zero absolute tolerance, and additionally demands an exact
   zero candidate-field Jacobian. No model, data, fitting setting, actual
   prediction-replay tolerance or risk threshold changed. All48 complete refits
   then reproduced exactly. The original failure and versions remain.

3. First allocation failed on a missing output parent directory. A tested outer
   readout entry creates that directory and invokes the unchanged registered
   engine. It cannot start training. No data, weights, solver budget, scientific
   choice or evaluation rule changed. Recovery completed all contexts before
   outcome readout. inference_storage_repair.md retains the failure.

## What Replay Establishes

Replay recalculates all model prediction arrays exactly, rechecks proposal and
independent choices exactly, verifies cached joint/uniform feasibility and
per-query matched counts, then recomputes every analysis metric exactly. Public
experiment files are hashed before and after the check and must be unchanged.
No fitting is called on this path; each replay creates a unique private receipt.

Replay does **not** rerun the one-second-bounded mixed-integer search. Its cached
choices are checked for feasibility, not certified as unique/global optima or
bitwise reproducible search trajectories. Exact computation replay does not
establish independent data validity, unseen-source performance or annotation-
generation causality. Full exact refits are a separate observed check above.

## Tests And Reporting Checks

The final eight-file focused suite passes 27 tests in 2.56s, including causal full-action
features, source-excluded fitting/resume, moment coherence, action-invariant
references and zero Jacobians, original support/NA behavior, joint controls,
same-count utility, role-aware reporting and selected-risk decomposition.
The entire historical suite was not rerun. These are engineering checks, not
independent scientific evidence. Actual data results are in analysis.json.

selection_calibration.csv recomputes original selected moment ratios from
float64 contract targets and matches the frozen evaluator. The registered
source/seed primary is independently re-aggregated exactly. Components and
conditional_risk_probe.json distinguish conditional versus composed targets.
The local two-panel figure was visually inspected; it explicitly shows both
utility and the failure to pass all72 required views. Private figures, raw
annotations, arrays and checkpoints are not published.

A second actual replay (PID80838, exit0, approximately14.082s) also reproduces
all predictions and metrics exactly after the report/diagnostic writers finish.
Both unique receipts remain private. Neither replay refits or reruns the timed
integer search, and neither is independent scientific confirmation.

## Scientific Status

Registered exploratory utility evidence is positive; complete original risk
and support fail. Deployment and the compiled v9 main PDF are unchanged. The
paper PDF SHA256 remains
`d9babe3ddbac858bcc1ac30e3ba98b332bed625c78ff9231d1d42a3e30b11772`.
New independent C/H confirmation and three forecast-teacher seeds are not run.
Original24-source-group main stays NA. No identity certification, metric/seconds,
true3D, foundation, human-gold, Stage5C or SMC claim is made.
