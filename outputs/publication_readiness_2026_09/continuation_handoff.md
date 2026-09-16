# Real-Experiment Continuation Handoff

Superseded for development on 2026-09-16: the user selected obs8/pred12 with
raw-frame t+50 supplemental and delegated the remaining research route. See
[the new decision](research_route_decision.md) and
`configs/m3w_8to12_development_v1.json`. Independent source/confirmation
requirements remain open. The earlier blocker is preserved below as history,
not a reason to ask the answered horizon question again.

## Current Executed State

Two real three-seed studies are complete: v1 coordinate MSE and v2 Smooth-L1.
Thirty neural fits completed 1,000 updates each; no active study child remains
after the runner completed all requested seeds. Checkpoints and per-fit heartbeats
are in the ignored `data/stage_cvpr2027_experiments/8to12_v1` and
`8to12_robust_v2` directories. Current main task is native obs8/pred12; raw t+50
supplement remains separate and not run.

Both studies select the CV floor in every seed. Mean uncontrolled primary gain
improves from -7.414% to -0.247%, but easy protection and joint-selection
contribution still fail. The robust candidate/floor oracle upper bound is only
0.231--0.351%; further threshold search cannot produce a 5% primary gain with
these fixed candidates. See [paired results](8to12_robust_v2/robust_loss_comparison.md)
and [failure conclusions](8to12_robust_v2/conclusions.md).

Next substantive work: investigate causal scale handling and a genuinely useful
public predictor, then run real deferral and realized-count-matched controls.
Do not build more status-only modules or repeat the same runtime probe. Preserve
all negative seeds. Native-coordinate improvements over CV alone do not beat
the stronger damped baseline consistently. Independent confirmation is still
missing, and no submission-ready or deployment claim is established.

MSE source snapshot: `707d4017`; robust source snapshot: `9e592089`.
Do not edit old protocol hashes to load changed implementations. Current code
runs v2 using the explicit command in the [runbook](local_create_runbook_zh.md).
CREATE access has not changed; no new remote job or absent-job claim. Stage5C and
SMC stay disabled. The research goal remains active, not completed.

## Historical Blocker Snapshot

Checked 2026-09-16. This is an execution blocker record, not research progress, a new gate or a completed experiment. The full CVPR research objective remains unfinished.

## Fresh Checks

- Current research commit before this handoff: `15cf26dd7cc71ffbd993f68fbb5dc00937f506cc`.
- Registered scientific protocol: `configs/m3w_independent_experiment.draft.json`; status `draft`, approval `null`.
- Protocol file SHA256: `1e6f401a5f75b22d031d6c1a2be80c3faece4ef16019d8a9778bf1cd6d2989d8`; unchanged.
- Nine recordings, six declared physical scenes; all nine retain `development_exposed` and `unassigned` roles.
- History length, prediction unit, horizon, primary metric and aggregation remain unspecified. Calibration risks, folds and seed registration are also incomplete.
- `.venv-pytorch/bin/python scripts/train_m3w_neural_cost_head.py --preflight-only` returned exit 2 with `Explicit protocol approval required` and `neural_cost_training_started: false`.
- A read-only host process check found no matching local M3W/stage/WorldCore/Pytest training command or CREATE SSH/sbatch command at inspection time. This is not a remote scheduler inspection or proof about unrelated processes. The first sandbox process check was denied; the authorized read-only check succeeded.
- The old draft binds earlier versions of `m3w_experiment_contract.py` and `m3w_joint_intervention.py`. Their hashes now differ after the documented repairs. These bindings must be reviewed and refreshed when a new protocol is frozen, not bypassed by changing only `status`.

## Existing Evidence, Not Recomputed Here

The [neural cost-head report](neural_cost_head/implementation_and_limits.md) records synthetic CPU/MPS fitting, recovery and comparison checks. The [support audit](risk_calibration/support_audit.md) records the lack of approved independent calibration scenes. Neither supplies a clean real forecasting result.

CREATE access previously failed with public-key authentication and a portal MFA notice; the project path is unresolved. No changed access condition was supplied, so the same failed connection was not retried. Remote jobs and artifacts remain unknown, not absent. No new job was submitted.

## Decisions Still Needed

Two previously raised scientific choices remain unanswered:

1. Use an 8-observation/12-prediction benchmark task as the main comparison, with raw-frame t+50 supplemental, or retain raw-frame t+50 as primary? Observation steps must not silently become seconds. The final approval must also specify primary ADE/FDE, aggregation, data roles and folds.
2. Prioritize an explicitly empirical, development-exposed matched-control study while seeking independent confirmation, or require acquisition of sufficient independently reviewed sites before the principal risk-control study? The former cannot produce a formal safety certificate or restore untouched-test status.

Recommended direction remains the benchmark-compatible empirical mechanism study first, with all previously inspected recordings labeled development material. This recommendation is **not approval**. The easy-degradation ceiling stays at the user's 2%; other statistical tolerances must not be inferred from that number.

New-source permission, annotation review and untouched-test eligibility are separate decisions. Approval of a task cannot clear the DUT duplicate-annotation quarantine or make its two sites into many independent scenes. CREATE access is optional for a small local comparison, not a prerequisite for answering the scientific choices.

## Resume Without Losing Work

After the decisions are supplied:

1. Write a new versioned protocol containing the approved choices and current source/code bindings. Preserve this draft and all historical exposure evidence.
2. Run the contract preflight, then a local cost-estimation pilot using the existing arm64 backend, explicit threads and zero DataLoader workers. Do not retrain models simply to repeat unchanged runtime checks.
3. Fit the prespecified causal/public forecasters and out-of-fold cost heads on assigned training data; compare deferral, ridge and neural heads on identical examples and candidate forecasts.
4. Run independent, scene-uniform and joint intervention controls, including the registered matched-count diagnostic. Select on development only, retain negative results and report the actual independent support.
5. Use approved independent calibration/confirmation data for any formal claim. If none are available, keep that claim unestablished rather than reusing development results as confirmation.

All commands, interfaces and known limits are in the [local/CREATE runbook](local_create_runbook_zh.md). Further architecture modules or repetitive reports will not remove the present decision blocker. No real model has been newly trained, deployed or claimed superior during this continuation. Stage5C and SMC remain disabled. The goal is blocked pending the missing scientific choices, not completed.
