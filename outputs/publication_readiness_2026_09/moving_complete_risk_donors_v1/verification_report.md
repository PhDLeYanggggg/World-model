# Complete-Risk Model Verification

## Material Passport

Experiment: `moving_complete_risk_donors_v1`. Role: exposed own-B development,
not a new independent test. Fresh preparation, fitting, exact refits, inference
and evaluation. Frozen forecasts, proposer and known-only controls are
cached_verified. No new C/H, deployment or forecaster-seed experiment.

## Real Execution

| Phase | PID | Observed exit | Seconds | Peak RSS bytes |
|---|---:|---:|---:|---:|
| Prepare | 71644 | 0 | 82.827966 | 3149840384 |
| Train 24 heads | 72064 | 0 | 292.336609 | 4659134464 |
| Complete 24 exact refits | 72750 | 0 | 296.148437 | 4690444288 |
| Freeze complete causal choices | 73273 | 0 | 1.711334 | 848625664 |
| Original full-policy evaluation | 73314 | 0 | 0.920200 | 1060159488 |

Native arm64 Torch, CPU4/inter-op1/workers0. Primary and repeat training each
complete 98,304 updates; all 24 full optimizer/model/trace refits are exact.
Checkpoint and explicit-resume code retain interruption state. Current runner
entry stages are single-use immutable receipts; resume is for interrupted
phases, not a second completed-phase receipt with a new PID.

## Review And Repair

A read-only source review finds no new query-source preprocessing leakage,
bound-target error or future-label input. Synthetic held-source perturbations
leave fitting tensors unchanged. Review is not real independent validation.

Two engineering gaps were addressed without changing frozen fitting definitions:
observed-versus-bound metrics are emitted by the reporting stage, and a separate
repeatable no-fit replay entry creates a new private execution receipt per run.
The original single-use runner behavior is disclosed rather than silently
changing a registered model/identity during training.

## Exact Replay

After reports stabilize, the repeatable replay succeeds twice:
PID73602 in 2.610848 s and PID73637 in 2.585461 s, both observed exit 0.
Each recomputes all causal risk predictions, choices and original complete
policy metrics exactly. Fitting is explicitly forbidden. Public files remain
unchanged during each replay. The second run preserves the first receipt.
This proves reproducible software output, not independent confirmation.

36 focused tests pass in 2.94 s. They cover target-role semantics, exact known
compatibility, source isolation, seed/resume behavior, support/NA, causal packet
parity, float/sign protection, reporting and repeatable no-fit replay. The full
historical test suite was not rerun. The unchanged compiled main paper SHA256 is
`d9babe3ddbac858bcc1ac30e3ba98b332bed625c78ff9231d1d42a3e30b11772`.

## Scientific Verdict

The original complete risk/support screen fails for every new policy. There
is no deployment upgrade, even though unknown admissions decrease. The separate
completion lower gain cannot replace the inconclusive paired contrast or failed
risk checks. Original rules, old failures and 24-group main NA remain.
Stage5C/SMC off; no metric/seconds, true-3D, foundation or submission claim.
