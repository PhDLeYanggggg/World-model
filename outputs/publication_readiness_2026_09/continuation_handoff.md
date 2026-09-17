# Real-Experiment Continuation Handoff

## Current State: v6 Primary and Supplements Complete (2026-09-17)

Both v6 families completed all registered seeds and fits: 24 forecasters at
10,000 updates, six neural cost heads at 1,000 updates and six ridge controls.
All six development choices are CV. Transformer primary gains are -5.736%,
-7.686%, -6.700%; EqMotion-K1 gains are -14.119%, -8.473%, -14.081%.
The complete pair is in `8to12_public_predictors_v6/paired_predictor_comparison.md`.
Do not restart the completed training or repeat its runtime pilot.

New error decompositions in each family's `error_scale.json` locate most positive
harm at the numerical past-scale floor, without removing rows or changing the
metric. Native Students03 EqMotion results beat the development-best causal
alternative by 0.94--2.32%, but Students01 and the primary result remain negative.
See `8to12_public_predictors_v6/conclusions.md` for the next falsifiable direction.

Both supplementary families completed all three seeds and twelve candidates.
Transformer has identical matched-count switch identities and zero ADE/FDE
difference. EqMotion changes only 76 repeated agent-query decisions, with tiny,
oppositely signed effects in seed29's two ridge policies and ten zero results.
Raw50 joint ADE gains are all negative. Both row replay audits match every
original fixed-forecast error and ordinary decision. No study process remains
active at the completion check. The last periodic heartbeat says running;
verified `completion.json` and the exited PID establish final completion.

Summary and replay reports are under each `8to12_<family>_v6_supplement` directory
in the publication output tree; large caches stay under ignored data. There is
no pending supplementary rerun. Next work must be a new prospective hypothesis,
not another status poll or threshold sweep. Main protocol digest remains
`53be3aafbda47ddf8d60891e779896f6685fcd59222a1fe4ea867e689f8914de`.
No independent-scene CI, deployment or submission-readiness claim. Stage5C/SMC
remain disabled; the research goal is active. The earlier snapshots below are
historical only, including resolved protocol-choice blockers and old run states.

## Historical v6 Launch Snapshot

The v5 seed29 full fit failed with nonfinite loss on both MPS and CPU, after
seed17 completed its four predictors, OOF heads and development evaluation.
v5 is incomplete, not a three-seed result; preserve its artifacts and the
`73b30e6a` source snapshot. Do not alter old protocol hashes to run new code.

Current runner: `scripts/run_m3w_conditioned_predictor_pair.py`, arm64 environment.
Protocol `configs/m3w_8to12_conditioned_context_v6.json`, digest
`53be3aafbda47ddf8d60891e779896f6685fcd59222a1fe4ea867e689f8914de`.
The only model factor is past-only common input conditioning, with predictions
restored before the unchanged loss/cost/evaluation scale. Decisions, source
caches, rows, seeds and 10,000-update budgets stay fixed. See the frozen v6
decision and pilot report. A real seed29 MPS 100-step pilot passed (13.01s),
and is reused/resumed when the runner reaches that seed; it is not a full fit.
48 tests passed; one optional MPS test skipped, separate real MPS run completed.

Active data directories: `8to12_eqmotion_v6` then `8to12_transformer_v6` under
`data/stage_cvpr2027_experiments`. Read their runner heartbeat and actual child
before resuming. Full/fold fits, OOF ridge, 1,000-update neural cost and development
evaluation run sequentially for seeds17/29/43. No bound-source edits during this
run. The summary and comparison helpers retain failed/negative seeds, all easy
metrics and actual runtimes. No v6 accuracy result yet; don't stop for slowness.

After both summaries, compare v6 metrics with `scripts/compare_m3w_public_predictors.py`.
One development site is not enough for a scene CI. EqMotion K=1 is not published
best-of-20. No independent confirmation, deployment, Stage5C or SMC claim.

## Historical v5 Run

Latest interruption: v5 EqMotion seed17 full/hold0/hold1 each completed 10,000.
hold2 failed at MPS step89 with float32 nonfinite output; outer pair runner exited.
The outer runner initially stopped. The explicit direct CPU resume of
seed17_hold2 subsequently completed all 10,000 updates in 2,093.36 seconds.
The original pair runner has now resumed, reusing all four completed predictors
and continuing OOF extraction. Its refreshed root heartbeat identifies the
current child. See
`8to12_public_predictors_v5/nonfinite_fit_diagnosis.md`. The MPS pre-failure
weights/batch are preserved locally. This is not the old OpenMP hang or a claim
of a root-cause fix. Do not skip folds; completed fits are hash-verified, not
retrained. Check actual current heartbeat/child before any additional restart.

Current source snapshot: `5e0f7be9`. Active command:
`scripts/run_m3w_continuous_predictor_pair.py` in arm64 `.venv-pytorch`.
Read `data/stage_cvpr2027_experiments/8to12_eqmotion_v5/runner_heartbeat.json`
and its indicated child log/checkpoint before any restart. It then runs the
matched Transformer at `8to12_transformer_v5`. Both register seeds 17/29/43,
full + three physical-fold predictors with 10,000 updates each, then OOF ridge
and 1,000-update neural gain/harm heads. Do not edit bound code while running.

v3 is an incomplete 200-update compute pilot. v4 completed seed17 full EqMotion
at 10,000 updates (1,151.87 s cumulative), then was deliberately halted before
development scoring after a source-context issue was established. Neither is
a completed three-seed comparison or accuracy result. v4 checkpoint preserved:
`data/stage_cvpr2027_experiments/8to12_eqmotion_v4/seed17_full/latest.pt`,
SHA256 `8525f739cb082a5475d8acbc933ae07abd5063bb4f397fe3d799954a6021f88d`.

The [Students01 row audit](students01_packaging_audit/audit.md) establishes exact
20-point prefix truncation and identity fragmentation, not a clock reset. All
17,820 packaged rows map to original timestamps/rounded coordinates; 3,993
short/tail rows and all 63 short tracks are absent. Their retention depended on
later availability, so past-only reader checks alone were insufficient for the
desired full-scene causal observation population. v5 uses the continuous local
source, keeping 415 original identities and 14,295 complete 8-to-12 windows.
Fit recordings, scientific choices and budgets remain unchanged. Historical
exposure is not cleared; annotation-generation causality is still unverified.

Protocol: `configs/m3w_8to12_continuous_context_v5.json`, digest
`24c0fb195ef76430f5d73ff06f1736afdef213d507c4ebd1d9cccdeba86d28e9`.
New fits restart rather than relabel old checkpoint identities. The completed
v4/v5 seed17 full fits were compared: all parameters, all 10,000 losses and
sampler order/cursor/RNG match exactly. See
`8to12_public_predictors_v5/unchanged_fit_replay.json`. This is one same-hardware
replay, not accuracy evidence; protocol identities/development populations differ.

EqMotion is the pinned public core, fixed K=1, not published best-of-20.
Unused-head execution was pruned with exact output/trainable-gradient matches
in CPU/MPS tests, without reducing selected-head model capacity or budget.
Runtime fallback cannot be silent. Current sequence is healthy local MPS/CPU,
not a new CREATE job; remote access conditions have not changed.

After both v5 summaries complete, run `scripts/compare_m3w_public_predictors.py`
with the two v5 metrics paths; retain every negative seed, easy error and native
strong-causal comparison. One University development scene cannot support a
scene CI. Source packaging changed, so v2-to-v5 aggregate changes are not a pure
model ablation. Update the run status/README/state with actual outputs, commit
only scoped light artifacts; unrelated staged fingerprint remains
`c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.

Goal still active: no submission readiness, no new deployment, no Stage5C/SMC.
Do not stop a healthy run because it is slow or replace it with status-only work.

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
