# Stage42 Reproducibility

## Environment

- training/eval scripts: `.venv-pytorch/bin/python` arm64
- tests: `python3 -m pytest tests`
- num_workers: `0` for torch data paths
- Stage5C executed: `False`
- SMC enabled: `False`

## Commands

```bash
.venv-pytorch/bin/python run_stage42_data_calibration.py
.venv-pytorch/bin/python run_stage42_external_validation.py
.venv-pytorch/bin/python run_stage42_full_waypoint_dynamics.py
.venv-pytorch/bin/python run_stage42_causal_ablation.py
.venv-pytorch/bin/python run_stage42_safety_floor.py
.venv-pytorch/bin/python run_stage42_paper_package.py
python3 -m pytest tests
```

## Source Labels

All Stage42 package claims use `fresh_run`, `cached_verified`, or `not_run`. Stage42-D explicitly does not relabel cached component ablations as fresh retraining.

<!-- STAGE42_AC_REFRESH:START -->
## Stage42-AC Latest Evidence Refresh

- source: `fresh_synthesis_from_stage42_wxyz_aa_ab_artifacts`
- scope: protected dataset-local raw-frame 2.5D paper package only.
- Stage42-AB is now included as auxiliary-head evidence.
- Auxiliary-head conclusion: mixed / partial; not a uniform main contribution claim.
- Stage5C and SMC remain disabled.

### Additional Command

```bash
.venv-pytorch/bin/python run_stage42_full_waypoint_auxiliary_ablation.py
python3 -m pytest tests/test_stage42_full_waypoint_auxiliary_ablation.py tests/test_stage42_paper_package_refresh.py
python3 -m pytest tests
```
<!-- STAGE42_AC_REFRESH:END -->

<!-- STAGE42_AJ_REFRESH:START -->
## Stage42-AJ Post-Repair Paper Package Refresh

- source: `fresh_synthesis_from_stage42_ad_to_ai_artifacts`
- scope: protected dataset-local raw-frame 2.5D paper package only.
- This refresh supersedes stale Stage42-AE limitations: horizon=25 harm, ETH_UCY t50/FDE, and TrajNet|100 easy safety were repaired by validation-only guards.
- t100 remains raw-frame diagnostic; metric/seconds/true-3D/foundation/Stage5C/SMC claims remain rejected.
- Future waypoints/endpoints remain labels/eval only, never inference inputs.

### Post-Repair Headline Metrics

- ADE all CI low: `0.085978`
- ADE t50 CI low: `0.058513`
- ADE t100 raw-frame diagnostic CI low: `0.068349`
- ADE hard/failure CI low: `0.090662`
- easy degradation CI high: `0.001168`
- FDE@50 CI low: `0.148230`

### Evidence Rows

| item | status | paper use | evidence |
| --- | --- | --- | --- |
| Stage42-AD calibration evidence refresh | `stage42_ad_calibration_evidence_refresh_pass` | data/calibration boundary | audited=7, files=1152, metric_allowed=False, seconds_allowed=False |
| Stage42-AF horizon25 validation-margin guard | `stage42_af_weak_slice_guard_repair_pass_with_eth_t50_limitation` | weak-slice safety repair | horizon25 -0.004781149088858072 -> 0.0; validation-only low-margin guard |
| Stage42-AG ETH_UCY t50/FDE source repair | `stage42_ag_eth_t50_fde_source_repair_pass` | domain t50/FDE lower-bound repair | ADE@50 low -0.013218100958604987 -> 0.002820688160982139; FDE@50 low -0.04199023614248535 -> 0.021040393452369632 |
| Stage42-AH post-repair claim matrix | `stage42_ah_post_repair_claim_refresh_pass` | claim matrix and remaining limitations | all_low=0.085258, t50_low=0.058513, hard_low=0.089767, easy_high=0.003348 |
| Stage42-AI TrajNet t100 easy-safety repair | `stage42_ai_trajnet_t100_safety_repair_pass` | raw-frame diagnostic t100 safety repair | TrajNet100 easy high 0.08498424090178214 -> 0.0; global t100 raw-frame low=0.068349 |

### Claim Boundary

- Supported: protected row-level full-waypoint raw-frame 2.5D world-state evidence with positive all/t50/hard/FDE@50 lower bounds and repaired t100 easy-safety diagnostic.
- Supported as non-harm only: horizon=25 floor/non-harm slices; do not call them positive dynamics contributions.
- Rejected: metric prediction, seconds-level horizon, true 3D, foundation model, Stage5C execution, SMC readiness, and ungated neural deployment.
<!-- STAGE42_AJ_REFRESH:END -->

<!-- STAGE42_CA_REFRESH:START -->
## Stage42-CA Post-BZ Paper Evidence Refresh

- source: `fresh_synthesis_from_stage42_by_bz_artifacts`
- scope: protected dataset-local raw-frame 2.5D paper package only.
- Stage42-BY repaired the protected t50 slices; Stage42-BZ adds bootstrap evidence.
- This is protected policy evidence under the Stage37/teacher floor, not floor-free neural world dynamics.
- t+50/t+100 remain raw-frame horizons; no global metric or seconds-level claim is allowed.
- Stage5C remains unexecuted and SMC remains disabled.

### Post-BZ Headline Evidence

- selected variant: `family_baseline_rel_only`
- internal validation group: `UCY::UCY/zara03/crowds_zara03.txt`
- robust t50 slices: `TrajNet|50, UCY|50`
- target union t50 CI: `[28.52%, 29.45%]`
- target union hard/failure CI low: `28.51%`
- target union easy degradation CI high: `-25.16%`
- bootstrap_n: `3000`

### Evidence Rows

| item | status | paper use | evidence |
| --- | --- | --- | --- |
| Stage42-BY protected t50 floor-relaxability repair | `stage42_by_t50_floor_relaxability_repair_pass` | point-estimate protected t50 slice repair | repaired=TrajNet|50, UCY|50; global t50=28.97%; easy=-37.05%; not floor-free neural |
| Stage42-BZ protected t50 bootstrap evidence | `stage42_bz_t50_repair_statistical_evidence_pass` | bootstrap-backed t50 statistical evidence | target union t50 CI=[28.52%, 29.45%]; hard CI low=28.51%; easy CI high=-25.16%; n=3000 |
| Stage42-BZ slice TrajNet|50 | `ci_positive_and_easy_safe` | slice-level t50 evidence | rows=9198; t50 CI=[29.80%, 30.67%]; easy CI high=-27.61%; switch=95.26% |
| Stage42-BZ slice UCY|50 | `ci_positive_and_easy_safe` | slice-level t50 evidence | rows=2340; t50 CI=[23.02%, 26.08%]; easy CI high=-8.16%; switch=65.00% |

### Claim Boundary

- Supported: protected t50 slice repair with bootstrap evidence for `TrajNet|50` and `UCY|50`.
- Still required: teacher/floor rollout context, protected safe-switch, train/internal-validation policy selection.
- Rejected: true 3D, foundation model, global metric prediction, seconds-level horizon, Stage5C execution, SMC readiness, and ungated/floor-free neural deployment.
<!-- STAGE42_CA_REFRESH:END -->

<!-- STAGE42_CW_RUNTIME_REPLAY_REFRESH:START -->
## Stage42-CW Runtime Replay Paper / Reproducibility Refresh

- source: `fresh_synthesis_from_stage42_cv_runtime_batch_replay`
- role: paper-ready deployment reproducibility evidence.
- Stage42-CV gate: `25 / 25`; verdict `stage42_cv_batch_runtime_replay_pass`.
- frozen policy hash: `4af6536f86499d5b39efa535bb81978398586d65746bb983571b642af7c92d59`.
- validation/test replay rows: `53256` / `55528`.
- exact runtime replay: validation `True`, test `True`.
- selected_xy / ADE / FDE max diff vs original CQ guard on test: `0.0` / `0.0` / `0.0`.
- test ADE vs endpoint-linear all/t50/t100 raw/hard: `1.77%` / `1.07%` / `3.48%` / `1.93%`.
- easy degradation: `0.25%`; switch rate: `16.96%`.
- near-collision@0.05 delta vs endpoint-linear: `-0.06%`; jagged-rate delta: `0.00%`.
- The guard's second proximity input is the validation-selected base composer candidate rollout group min-distance, not future labels.
- This refresh does not create a new metric/seconds/3D/foundation claim; it only strengthens deployable policy reproducibility under protected dataset-local/raw-frame 2.5D boundaries.
- Stage5C remains unexecuted and SMC remains disabled.
<!-- STAGE42_CW_RUNTIME_REPLAY_REFRESH:END -->

<!-- STAGE42_CX_EVIDENCE_PROVENANCE:START -->
## Stage42-CX Evidence Provenance / Command Matrix

- source: `fresh_evidence_provenance_from_stage42_artifacts`
- This section lists high-value Stage42 evidence artifacts, their source labels, gates, runners, and worktree caveats.
- A worktree caveat is not hidden evidence; it means the current file differs from committed HEAD or is untracked.

| claim area | source label | gate | runner | worktree caveat |
| --- | --- | --- | --- | ---: |
| `data_calibration` | `fresh_run` | `7/7` | `run_stage42_data_calibration.py` | `False` |
| `external_validation` | `fresh_run` | `10/10` | `run_stage42_external_validation.py` | `False` |
| `full_waypoint_dynamics` | `fresh_run` | `12/12` | `run_stage42_full_waypoint_dynamics.py` | `False` |
| `causal_ablation` | `fresh_run` | `12/12` | `run_stage42_causal_ablation.py` | `False` |
| `safety_floor` | `fresh_run` | `12/12` | `run_stage42_safety_floor.py` | `False` |
| `paper_package` | `fresh_run` | `12/12` | `run_stage42_paper_package.py` | `False` |
| `strict_time_geometry_calibration` | `fresh_run` | `13/13` | `run_stage42_source_time_geometry_calibration.py` | `False` |
| `metric_time_claim_guard` | `fresh_run` | `11/11` | `run_stage42_metric_time_claim_guard.py` | `False` |
| `source_terms_validation` | `fresh_run` | `11/11` | `run_stage42_source_terms_confirmation_validator.py` | `False` |
| `context_contribution_forensics` | `fresh_run` | `13/13` | `run_stage42_context_contribution_forensics.py` | `False` |
| `goal_scene_gated_expert` | `fresh_run` | `10/10` | `run_stage42_goal_scene_gated_expert.py` | `False` |
| `neighbor_interaction_gated_expert` | `fresh_run` | `11/11` | `run_stage42_neighbor_interaction_gated_expert.py` | `False` |
| `common_validation_bridge_shape_composer` | `cached_verified` | `14/14` | `run_stage42_common_validation_bridge_shape_composer.py` | `False` |
| `composer_safety_bootstrap` | `fresh_run` | `14/14` | `run_stage42_common_validation_composer_safety.py` | `False` |
| `proximity_aware_composer_guard` | `fresh_run` | `19/19` | `run_stage42_proximity_aware_composer_guard.py` | `False` |
| `proximity_guard_ablation` | `fresh_run` | `19/19` | `run_stage42_proximity_guard_ablation.py` | `False` |
| `frozen_proximity_guard_policy` | `fresh_run` | `25/25` | `run_stage42_freeze_proximity_guard_policy.py` | `False` |
| `frozen_policy_replay` | `fresh_run` | `30/30` | `run_stage42_replay_proximity_guard_policy.py` | `True` |
| `runtime_policy_api` | `fresh_run` | `19/19` | `run_stage42_runtime_proximity_guard_policy.py` | `False` |
| `batch_runtime_replay` | `fresh_run` | `25/25` | `run_stage42_batch_replay_proximity_guard_policy.py` | `True` |
| `runtime_replay_paper_refresh` | `fresh_run` | `25/25` | `run_stage42_runtime_replay_paper_refresh.py` | `False` |
| `group_consistency_full_waypoint_repair` | `fresh_run` | `17/17` | `run_stage42_group_consistency_full_waypoint_repair.py` | `False` |
| `frozen_group_consistency_policy` | `fresh_run` | `22/22` | `run_stage42_freeze_group_consistency_policy.py` | `False` |
| `group_consistency_policy_replay` | `fresh_run` | `34/34` | `run_stage42_replay_group_consistency_policy.py` | `True` |
| `group_consistency_runtime_policy` | `fresh_run` | `30/30` | `run_stage42_group_consistency_runtime_policy.py` | `True` |
<!-- STAGE42_CX_EVIDENCE_PROVENANCE:END -->

<!-- STAGE42_DJ_FROZEN_GROUP_CONSISTENCY_POLICY:START -->
## Stage42-DJ Frozen Group-Consistency Full-Waypoint Policy

- source: `fresh_policy_freeze_from_stage42_di`
- role: freeze the Stage42-DI promoted group-consistency full-waypoint repair as a reproducible policy artifact.
- repair uses predicted rollout geometry and source/frame/horizon group keys only; future waypoints remain labels/eval only.
- policy artifact: `outputs/stage42_long_research/frozen_group_consistency_full_waypoint_policy_stage42_policy.json`
- policy hash: `617ef9952b1439f3678318129a4979c7a171f2ba882742cd18acd46c5ae92141`
- test vs train-horizon causal floor ADE: all `24.72%`, t50 `22.36%`, t100 raw `14.35%`, hard `23.89%`, easy `-25.63%`.
- delta vs Stage42-AM all/t50/hard: `0.14%` / `0.35%` / `0.14%`.
- near@0.05 base/final/floor: `1.94%` / `1.38%` / `2.24%`.
- claim boundary: still protected dataset-local/raw-frame 2.5D; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.
<!-- STAGE42_DJ_FROZEN_GROUP_CONSISTENCY_POLICY:END -->

<!-- STAGE42_DK_GROUP_CONSISTENCY_POLICY_REPLAY:START -->
## Stage42-DK Group-Consistency Policy Replay

- source: `fresh_replay_from_frozen_group_consistency_policy_artifact`
- role: replay the Stage42-DJ frozen group-consistency full-waypoint policy artifact against Stage42-DI/DJ source evidence.
- replay performs no retraining, no threshold reselection, and no test tuning.
- policy artifact: `outputs/stage42_long_research/frozen_group_consistency_full_waypoint_policy_stage42_policy.json`
- policy hash: `617ef9952b1439f3678318129a4979c7a171f2ba882742cd18acd46c5ae92141`
- gate: `34 / 34`; verdict `stage42_dk_group_consistency_policy_replay_pass`.
- replayed ADE vs train-horizon causal floor: all `24.72%`, t50 `22.36%`, t100 raw `14.35%`, hard `23.89%`, easy `-25.63%`.
- replayed near@0.05 base/final/floor: `1.94%` / `1.38%` / `2.24%`.
- claim boundary: still protected dataset-local/raw-frame 2.5D; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.
<!-- STAGE42_DK_GROUP_CONSISTENCY_POLICY_REPLAY:END -->

<!-- STAGE42_DL_GROUP_CONSISTENCY_RUNTIME_POLICY:START -->
## Stage42-DL Group-Consistency Runtime Policy API

- source: `fresh_runtime_api_from_frozen_group_consistency_policy_artifact`
- role: expose Stage42-DJ/DK frozen group-consistency full-waypoint repair as a callable runtime policy.
- real batch replay uses reconstructed Stage42-DI source-level test rows and checks exact selected trajectory replay.
- policy artifact: `outputs/stage42_long_research/frozen_group_consistency_full_waypoint_policy_stage42_policy.json`
- policy hash: `617ef9952b1439f3678318129a4979c7a171f2ba882742cd18acd46c5ae92141`
- gate: `30 / 30`; verdict `stage42_dl_group_consistency_runtime_policy_pass`.
- replayed ADE vs train-horizon causal floor: all `24.72%`, t50 `22.36%`, t100 raw `14.35%`, hard `23.89%`, easy `-25.63%`.
- replayed near@0.05 base/final/floor: `1.94%` / `1.38%` / `2.24%`.
- claim boundary: still protected dataset-local/raw-frame 2.5D; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.
<!-- STAGE42_DL_GROUP_CONSISTENCY_RUNTIME_POLICY:END -->
