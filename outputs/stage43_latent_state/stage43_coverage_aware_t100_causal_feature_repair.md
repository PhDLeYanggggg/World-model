# Stage43-CK Coverage-Aware T100 Causal Feature Repair

- source: `fresh_stage43_ck_coverage_aware_t100_causal_feature_repair`
- result_source: `fresh_stage43_ck_coverage_aware_t100_causal_feature_repair`
- gate: `18 / 18`
- verdict: `stage43_ck_t100_causal_feature_repair_pass_keep_ci_floor`
- deploy t100 causal specialist: `False`
- checkpoint committed: `False`

## Why This Stage Exists

- Stage43-CJ trained a t100 specialist but did not deploy it.
- During follow-up audit, CJ's specialist feature set was found to include true-error diagnostics derived from future waypoints.
- Because CJ kept the Stage43-CI floor, the deployable policy was not contaminated.
- CK replaces that diagnostic with a causal-only t100 specialist trial and marks CJ as non-admissible for no-leakage t100 specialist evidence.

## Prior CJ Audit

- prior CJ verdict: `stage43_cj_t100_long_horizon_specialist_pass_keep_ci_floor`
- prior CJ deployed t100 specialist: `False`
- label-derived features found: `['cg_candidate_ade', 'cg_candidate_fde', 'floor_ade', 'floor_fde']`
- deployment contamination: `False`

## Causal Feature Contract

- feature dim: `214`
- included: causal CE feature vector, floor rollout delta, CG latent waypoint candidate, CG latent state, CG gain/harm/failure/density heads.
- excluded: true candidate/floor ADE/FDE, oracle errors, future endpoint, future waypoints as input.

## Claim Boundary

- Not true 3D.
- Not a foundation world model.
- Dataset-local/raw-frame 2.5D evidence only.
- No metric or seconds-level claim.
- Stage5C not executed.
- SMC not enabled.

## Deployed Test Metrics

- full-waypoint ADE improvement: `52.03%`
- t50 full-waypoint ADE improvement: `31.13%`
- t100 raw-frame diagnostic: `0.00%`
- hard/failure improvement: `50.48%`
- easy degradation: `0.00%`
- switch rate: `69.09%`

## Delta Vs Stage43-CI Floor

- all delta: `0.00%`
- t50 delta: `0.00%`
- t100 delta: `0.00%`
- hard/failure delta: `0.00%`
- easy degradation delta: `0.00%`

## Validation-Selected Causal Specialist Diagnostic

- all: `52.03%`
- t50: `31.13%`
- t100 raw-frame diagnostic: `0.00%`
- hard/failure: `50.48%`
- easy degradation: `0.00%`

## Raw Causal Candidate T100 Diagnostic

- rows: `8443`
- t100 full-waypoint ADE improvement: `-3.84%`
- t100 endpoint FDE improvement: `-2.83%`
- t100 easy degradation: `22.49%`

## Bootstrap CI For Deployed Policy

- bootstrap n: `2000`
- all CI: `[51.64%, 52.40%]`
- t50 CI: `[30.47%, 31.80%]`
- t100 CI: `[0.00%, 0.00%]`
- hard/failure CI: `[50.04%, 50.92%]`

## Interpretation

The causal-only t100 specialist is the admissible no-leakage diagnostic. It still does not produce a positive/easy-safe t100 switch, so Stage43-CI remains the deployed t100 floor.

## Gate

| gate | passed |
| --- | --- |
| `prior_cj_leakage_audited` | `True` |
| `prior_cj_not_deployed_or_flagged` | `True` |
| `ci_precondition_passed` | `True` |
| `fresh_torch_training` | `True` |
| `checkpoint_not_committed` | `True` |
| `causal_only_features` | `True` |
| `validation_selected` | `True` |
| `no_test_threshold_tuning` | `True` |
| `future_waypoints_label_only` | `True` |
| `no_future_endpoint_or_central_velocity` | `True` |
| `no_test_goal_or_stat_leakage` | `True` |
| `all_still_positive` | `True` |
| `t50_not_destroyed` | `True` |
| `hard_failure_still_positive` | `True` |
| `easy_preserved` | `True` |
| `t100_result_honest` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
