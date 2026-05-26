# Stage42 A-Journal Gap Analysis

## Current Position

Stage42 is strong enough to support a serious protected 2.5D external world-state dynamics manuscript draft. It is not yet enough for a broad true-3D/foundation/world-model claim.

## What Is Already Paper-Usable

- Fresh source-level external validation.
- Fresh full-waypoint all-agent world-state evaluation.
- Fresh safety-floor study showing why ungated neural cannot be deployed.
- Fresh Stage42-G Phase1 retrained external selector ablations for history, neighbor/interaction, goal/scene, domain expert, safe-switch, and teacher-floor proxy variants.
- Fresh Stage42-H causal temporal sequence ablation showing that history tokens are strongly positive when encoded as a sequence rather than flattened into a ridge-selector feature vector.
- Fresh Stage42-I sequence-to-full-waypoint experiment showing that causal history gives a small positive full-waypoint contribution, while un-gated static/context features currently hurt protected ADE.
- Fresh Stage42-J static-gated full-waypoint repair showing that validation-selected partial-static experts convert the Stage42-I negative full model into positive ADE/FDE full-waypoint evidence while preserving easy cases.
- Fresh Stage42-K static-gated checkpoint training showing that a learned static gate/dropout can be trained directly into a checkpoint and improve over Stage42-I full static+sequence while preserving easy cases.
- Fresh Stage42-L horizon-aware static gate repair showing that t+50-specific gate conditioning fixes the Stage42-K ADE t50 sign while preserving easy cases.
- Fresh Stage42-M policy-distillation negative result showing that coarse domain/horizon alpha distillation is insufficient; row-level gain/harm supervision is needed.
- Fresh Stage42-N row-level gain/harm static-gate pilot showing that row-level alpha supervision improves all/hard but still fails t+50, so alpha-style gate distillation alone is insufficient.
- Fresh Stage42-O explicit gain/harm selector showing that row-level switch/gain/harm prediction improves all/hard and uses train-only normalization, but still does not pass ADE t50.
- Fresh Stage42-P t+50-specific gain/harm selector showing that t+50-weighted train/val supervision repairs the mean ADE t50 sign while preserving all/hard/easy.
- Clear claim boundaries and no-leakage policy.

## What Is Not Yet Strong Enough

- Full retrained ablation for every named component: Stage42-G/H cover key feature/safety selector and causal sequence-history ablations, but JEPA, full Transformer, endpoint-bridge, and full-waypoint-shape retraining remain open.
- Full sequence-to-waypoint deployment: Stage42-L repairs the fresh checkpoint t50 sign, but it still underperforms the Stage42-J policy-level gate. A stronger paper claim still needs distillation of Stage42-J's domain/horizon expert selection into a fresh checkpoint, longer training, or bootstrap over the improved checkpoint.
- Policy distillation and row selector: Stage42-M shows that distilling only slice-level static alpha can improve FDE t50 but harms ADE t50. Stage42-N shows that row-level alpha/gain/harm supervision improves all/hard but still harms ADE t50. Stage42-O shows that an explicit gain/harm selector improves all/hard more cleanly under train-only normalization, but ADE t50 is still slightly negative. Stage42-P repairs the mean ADE t50 sign with t+50-specific weighting, but the 3-seed t50 CI low remains negative, so it still needs statistical strengthening before becoming a paper-level stable t50 claim.
- Metric/time-calibrated pedestrian benchmark claims.
- External expansion beyond the current converted top-down state with independent legal datasets.
- Floor-free or partially floor-free neural deployment that preserves proximity/collision safety.
- Strong JEPA/full-Transformer positive contribution claim; current evidence favors protected bounded dynamics and causal sequence-history modeling over pure JEPA/Transformer.

## Shortest Next Path

1. Add more seeds/bootstrap and per-domain horizon calibration for Stage42-P, because it passes the mean t50 gate but still has negative 3-seed t50 CI low.
2. Run Stage42-G/H Phase2 true retrained ablations for no-JEPA, no-Transformer, no-endpoint-bridge, and no-full-waypoint-shape with bootstrap or three seeds; Stage42-H has repaired the history-token question with an actual sequence model, so the next ablation priority is full Transformer/JEPA/full-waypoint-shape rather than flattened-history.
3. Add one more legally verified external top-down pedestrian/drone dataset or a stronger held-out source split.
4. Build a proximity-safe internal self-gate that reduces teacher-floor dependence without increasing collision/proximity risk.
5. Obtain verified homography/FPS/stride for at least one pedestrian subset, or keep all claims raw-frame/dataset-local.

## Absolute Non-Claims

- Not true 3D.
- Not foundation.
- Not metric/seconds-level pedestrian prediction.
- Not Stage5C or SMC.
- Not ungated neural deployment.

<!-- STAGE42_AC_REFRESH:START -->
## Stage42-AC Latest Evidence Refresh

- source: `fresh_synthesis_from_stage42_wxyz_aa_ab_artifacts`
- scope: protected dataset-local raw-frame 2.5D paper package only.
- Stage42-AB is now included as auxiliary-head evidence.
- Auxiliary-head conclusion: mixed / partial; not a uniform main contribution claim.
- Stage5C and SMC remain disabled.

### Updated A-Journal Gap

- Need stronger uniformly positive scene/goal/interaction evidence or a cleaner theoretical framing that treats these heads as safety/diagnostic auxiliaries rather than core dynamics modules.
- Need metric/time calibration or continue strict raw-frame language.
- Need broader external legal datasets and stronger floor-reduction evidence before claiming foundation-track generalization.
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

<!-- STAGE42_CB_REFRESH:START -->
## Stage42-CB Source Robustness Caveat

- source: `fresh_stage42_cb_t50_source_robustness_audit`
- verdict: `stage42_cb_t50_source_robustness_pass_with_source_diversity_limit`
- robust major-source slices: `TrajNet|50`, `UCY|50`
- concentration-limited slices: `TrajNet|50`, `UCY|50`
- broad source-level generalization claim allowed: `false`

Stage42-CB strengthens the honesty of the t50 claim: BY/BZ t50 gains are positive on the available major sources, but source diversity is still limited. `TrajNet|50` is dominated by `students003` and `UCY|50` currently has a single test source. This supports a paper claim of protected t50 repair with source-concentration caveat, not a broad source-level generalization claim. The shortest path remains acquiring or legally converting more independent t50-capable external top-down sources.
<!-- STAGE42_CB_REFRESH:END -->

<!-- STAGE42_CC_REFRESH:START -->
## Stage42-CC Independent T50 Source Inventory

- source: `fresh_stage42_cc_independent_t50_source_inventory`
- verdict: `stage42_cc_independent_t50_source_inventory_pass`
- scanned files: `93`
- t50-capable files: `10`
- unused independent ready-to-claim t50 sources: `0`
- alternate current-source candidates: `4`
- diagnostic/synthetic t50 candidates: `1`
- source-diversity repair ready: `false`

Stage42-CC confirms that the Stage42-CB source-diversity caveat remains active. The local inventory contains t50-capable files, but the conservative audit does not count any as an unused independent ready-to-claim real external source. Four files are alternate/current-source representations useful for split-rebuild diagnostics, and one file is synthetic/diagnostic. Therefore BY/BZ t50 repair remains paper-usable only with a source-concentration caveat; broad source-level generalization still requires legally enabled independent top-down pedestrian sources followed by conversion, no-leakage audit, validation-only selection, and final test.

### Updated Gap After CC

- Need at least one new independent legal t50-capable external source for source-diversity repair.
- Do not count alternate representations, registry-only paths, or synthetic/diagnostic rows as converted/evaluated external evidence.
- The current A-journal package can claim protected major-source robustness, not broad source-level generalization.
<!-- STAGE42_CC_REFRESH:END -->

<!-- STAGE42_CD_REFRESH:START -->
## Stage42-CD Source Diversity Acquisition Package

- source: `fresh_stage42_cd_source_diversity_acquisition_package`
- verdict: `stage42_cd_source_diversity_acquisition_package_pass`
- official targets: `5`
- auto-download targets: `0`
- manual/terms targets: `4`
- local path families found: `4`
- converted datasets now: `0`
- source-diversity repair ready now: `false`
- broad source-level generalization claim allowed: `false`

Stage42-CD converts the CC source-diversity blocker into an actionable acquisition package. It identifies UCY Crowd Data, ETH/BIWI, TrajNet++, OpenTraj toolkit/reference, and an additional legal top-down pedestrian/drone source as the next acquisition/conversion targets. It intentionally performs no automatic download and makes no conversion claim, because the priority targets either require manual terms/path verification or are toolkit/challenge references rather than independent permission to use data.

### Updated Gap After CD

- The next paper-critical data action is not another source-concentrated t50 report; it is legal independent source acquisition plus conversion/no-leakage/source-CV.
- Existing local paths are useful, but current evidence still cannot claim broad source-level generalization.
- A future Stage42 source-diversity repair must start from a user-provided legal local path or verified official terms.
<!-- STAGE42_CD_REFRESH:END -->

<!-- STAGE42_CE_REFRESH:START -->
## Stage42-CE Source Diversity Conversion Preflight

- source: `fresh_stage42_ce_source_diversity_conversion_preflight`
- verdict: `stage42_ce_source_diversity_conversion_preflight_pass`
- gates: `12 / 12`
- targets checked: `5`
- targets with local path: `4`
- targets with schema possible: `4`
- targets with t50 files: `3`
- targets with t100 files: `3`
- independent t50 candidates ready to claim: `0`
- source-CV-ready targets now: `0`
- converted datasets now: `0`
- evaluated datasets now: `0`
- source-diversity repair ready now: `false`

Stage42-CE is a useful engineering preflight, not a data-conversion success claim. It shows that UCY/ETH/BIWI/TrajNet/OpenTraj-related local paths are present and often parseable, but the legal/source-identity blockers remain: UCY/ETH/TrajNet still need official terms/path verification, OpenTraj is a toolkit/root scan rather than blanket independent data permission, and no target produces an unused independent ready-to-claim t50 source.

### Updated Gap After CE

- The source-diversity blocker is now narrower: local parseability exists, but legal permission, source identity, and source-CV readiness are still missing.
- Do not count Stage42-CE as converted data, external evaluation, or broad source-level generalization.
- The next valid repair step is legal terms/path verification followed by actual conversion, no-leakage audit, validation-only source-CV selection, and final test once.
<!-- STAGE42_CE_REFRESH:END -->

<!-- STAGE42_CF_REFRESH:START -->
## Stage42-CF Source Conversion Legal Gate

- source: `fresh_stage42_cf_source_conversion_legal_gate`
- verdict: `stage42_cf_source_conversion_legal_gate_pass`
- gates: `13 / 13`
- targets checked: `5`
- local paths present: `4`
- schema possible targets: `4`
- targets with t50 files: `3`
- targets with t100 files: `3`
- source-CV-ready targets now: `0`
- conversion allowed now: `0`
- converted datasets now: `0`
- evaluated datasets now: `0`

Stage42-CF turns the Stage42-CE preflight into an enforceable legal/source-identity gate. It deliberately permits no conversion at this time: local parseability is not permission, OpenTraj toolkit/root scans are not independent data rights, and UCY/ETH/TrajNet still need explicit official terms/path verification before any conversion can be counted. The generated terms-confirmation JSON is a checklist only.

### Updated Gap After CF

- The next source-diversity step is no longer ambiguous: fill legal confirmation from official terms, isolate independent source identity, then run actual conversion/no-leakage/source-CV/final test.
- Current paper evidence remains strong for protected dataset-local/raw-frame 2.5D dynamics, but broad source-level generalization is still blocked.
- Do not cite Stage42-CF as data conversion, external validation, metric/time calibration, true-3D evidence, Stage5C readiness, or SMC readiness.
<!-- STAGE42_CF_REFRESH:END -->
