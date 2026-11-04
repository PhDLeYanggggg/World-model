# Amplitude-Conditioned Intervention Cost Heads

## Intended Use

Controlled development of a causal forecast-admission policy on fit-admitted
MovingDroneCrowd++ source data. Not deployed, not a calibrated risk guarantee,
not a newly trained trajectory forecaster, not a foundation or true 3D model.

## Model And Actions

- 36 fresh real Torch query-context heads; two arms across 18 producer/critic/
  held orientations, with three actual seeds 17, 29 and 43.
- 28,229 parameters per head; width 64; 428 causal inputs, with the 48
  shared-motion fields zeroed in both arms.
- Same seven candidate inputs: physical floor and fractions
  {1/32, 1/16, 1/8, 1/4, 1/2, 1} of the frozen neural/floor difference.
- Binary-only supervises/permits only the full neural action. Amplitude-bank
  supervises/permits all six nonzero actions. A same-weight binary deployment
  is a secondary action-availability control.
- Outputs are expected benefit, positive harm, shared reference error, shared
  easy reference error and easy harm. The last two references are not scaled
  by intervention size; original CV defines the easy event.
- Actual candidate-to-floor discrepancy bounds constrain both new arms.
  Costs are recomputed for changed paths, not transported from the old action.

## Data And Supervision

195,286 past-eligible windows from the unchanged 64-recording fit admission.
58 recordings in 23 source groups support the native 8/12 task. The original
24-group primary is not estimable. Overlapping windows are not independent
replicates, and source-group identity is not verified geographic independence.

Per pipeline, forecaster-fitting, cost-fitting and held source groups remain
disjoint. Normalizers and easy/hard cutoffs are inherited from their original
permitted fitting roles. No future endpoint, future-validity mask, oracle
action, central velocity, test normalization or test-endpoint goal is accepted
by the inference schema. Future outcomes/masks are supervision and evaluation
labels only. The fitting diagnostic constructs training labels before discarding
the training payload, then computes causal choices and diagnostic scores;
it does not claim labels were never computed earlier in that routine.

Wholly unknown futures contribute conservative H/EH bounds and no observed
R/ER mass, not invented zero future errors. Available-label ADE for partial
futures does not identify error on their unobserved steps. No new data roles,
image inputs, calibration or final-confirmation access are admitted.

## Training And Recovery

Native arm64 CPU, 4 compute threads, 1 interop thread, workers 0, no resource
probing. AdamW lr 0.001, gradient clip 5, source-balanced query batches of 72,
2,000 primary updates per head and at least four full query cycles.
All actual heads complete at least 25 query visits. Total: 72,000 primary
updates plus two separate complete 2,000-update verification refits.
Pilot work is separate. Atomic checkpoints and exact same-binding resume
are supported. Training took 3,025.522778 s, peak RSS 8,712,224,768 bytes.

Both complete first-job refits and all 18 paired initial states, draw streams
and fitting identities agree exactly. Held evaluation took 910.433301 s,
peak RSS 6,212,059,136 bytes. The phase completion receipt and final complete
log were observed; the original shell-session handle was lost in truncated
tool output, so a separately observed shell exit code is not asserted for
that phase. Separate full replay completes with observed exit 0 as PID 25603
in 199.648098 s, peak RSS 6,264,569,856 bytes. All new cost outputs and policy
scalars match exactly; the 54 saved solver receipts retain their fixed budget
and pass binding/feasibility checks.
Weights and raw/intermediate data remain private and are not committed.

## Evaluation And Non-Deployment

All 138 supported source/producer/seed views and 348 repeated recording views
remain. Six policy controls are retained for both new heads, same-weight
binary deployment and cached legacy reference. Joint allocation pools the
predicted 2% all/easy budgets; it does not optimize geometric collision costs.
The fixed solver budget is 0.25 s/query. Timeouts retain the predicted-feasible
independent parent; this is not true-risk support or an exact optimum claim.
Verification checks saved bindings/feasibility, not bitwise MILP reoptimization.

Nominal intervals use 3,000 source-group resamples after producer/seed averaging.
They are development intervals, not independent confirmation or multiple-search
correction. Empty same-count utility contrasts are not estimable, not safe zero.
Equal switch counts do not imply equal amplitude or path-change mass.

The fixed joint-free paired completion-lower effect is -1.158427 pp
[-2.906426, +0.218829]. Only 10/348 amplitude-bank recording views meet the
original full risk contract. The primary and complete risk requirements fail.
There is no deployment promotion. See the full readout and failure analysis
for the small known-label same-weight secondary effect and its limitations.

## Reproduction

Use the ordered commands and recovery rules in `operation_zh.md`. The registered
definition was committed before fitting; no frozen training or scientific
evaluation definition changed after registration. Reports/tests outside that
definition add diagnostics, not new model selection or a changed primary.
Exact weights, data admission and forecast lineage are required locally;
the public code alone does not provide third-party data or private weights.

Current results remain image-plane trajectory research within a 2.5D scaffold.
Pixel and native annotation-step units are not metric or verified time.
Stage5C execution and SMC remain off; the project is not submission-ready.
