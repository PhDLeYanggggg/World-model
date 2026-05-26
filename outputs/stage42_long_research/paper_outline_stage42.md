# Stage42 Paper Outline

## Working Title

M3W-Neural v1: Protected Multi-Agent Full-Waypoint World-State Dynamics for External Top-Down Trajectory Benchmarks

## Core Thesis

A protected neural dynamics model with a Stage37/teacher safety floor can deliver positive external raw-frame 2.5D multi-agent trajectory/world-state improvements while preserving easy cases and proximity safety. The current evidence supports a protected 2.5D world-state candidate, not a true 3D, metric, seconds-level, or foundation model.

## Main Contributions

1. Source-level external validation over ETH_UCY, TrajNet, UCY/OpenTraj-derived state with protected composite-tail dynamics.
2. Full-waypoint sequence evaluation over all active agents, beyond endpoint-only linear bridge diagnostics.
3. Safety-floor analysis showing ungated neural is high-lift but unsafe, while protected safe-switch remains deployable.
4. Evidence ledger with fresh_run/cached_verified/not_run boundaries and no future/test leakage claims.

## Paper Structure

1. Introduction and problem setting
2. Related work placeholder: trajectory forecasting, world models, JEPA, safe fallback policies
3. Method: causal features, Stage37 floor, composite-tail bounded dynamics, full-waypoint model
4. Data and calibration: dataset-local raw-frame external benchmark, no metric/time overclaim
5. Experiments: external validation, full-waypoint dynamics, safety-floor study
6. Ablations and negative evidence
7. Limitations and A-journal gap
8. Reproducibility checklist

## Claim Matrix

| claim | status | evidence |
| --- | --- | --- |
| protected external raw-frame 2.5D world-state dynamics improves over Stage37/strongest floor | `supported` | external all=0.2103, t50=0.1365, hard=0.2038, easy=-0.1451 |
| full-waypoint sequence dynamics exists beyond endpoint-only linear bridge | `supported_but_protected` | full-waypoint ADE all=0.1858, t50=0.1480, t100diag=0.2286, positive_domains=['ETH_UCY', 'TrajNet'] |
| ungated neural can replace safety floor | `rejected` | ungated easy degradation=1.2459; safety conclusion=teacher_floor_required_for_current_deployment |
| metric or seconds-level pedestrian world model | `not_supported` | global_metric=False, global_seconds=False |
| true 3D or foundation world model | `not_supported` | all Stage42 claim boundaries keep true_3d=false and foundation_world_model=false |
| scene/goal/interaction/history/neighbor contributions are proven | `partially_supported` | ablation coverage=True; all Stage42-D component retraining=False |
| A-journal submission candidate | `candidate_package_not_final_claim` | A-E evidence is organized and strong for a protected 2.5D paper; full retrained ablation, metric/time calibration, independent external expansion, and floor-free safety remain gaps. |

<!-- STAGE42_AC_REFRESH:START -->
## Stage42-AC Latest Evidence Refresh

- source: `fresh_synthesis_from_stage42_wxyz_aa_ab_artifacts`
- scope: protected dataset-local raw-frame 2.5D paper package only.
- Stage42-AB is now included as auxiliary-head evidence.
- Auxiliary-head conclusion: mixed / partial; not a uniform main contribution claim.
- Stage5C and SMC remain disabled.

### Latest Claim Boundary

The paper-ready scope remains a protected 2.5D raw-frame world-state candidate. Stage42-AB adds a useful negative/mixed result: interaction/occupancy/physical auxiliary losses help t50 slightly but do not improve all/hard ADE uniformly.
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
