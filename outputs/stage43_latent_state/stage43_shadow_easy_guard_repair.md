# Stage43-CC Shadow Easy Guard Repair

- source: `fresh_stage43_cc_shadow_easy_guard_repair`
- result_source: `fresh_shadow_validation_easy_guard_repair`
- verdict: `stage43_cc_shadow_easy_guard_shadow_safe_test_mismatch`
- gate: `12 / 13`
- selected policy: `base_threshold_only`
- deployable policy changed: `False`

## Shadow Validation

- calibration rows: `71074`
- shadow holdout rows: `30372`
- validation source families: `['biwi', 'hotel', 'students']`
- test global-unsupported families: `{'pets': 1926, 'zara': 9540}`
- test domain-unsupported families: `{'ETH_UCY|students': 70585, 'TrajNet|pets': 1926, 'UCY|zara': 9540}`
- shadow all improvement: `0.1321`
- shadow t50 improvement: `-0.0079`
- shadow hard/failure improvement: `0.1690`
- shadow easy degradation: `0.0010`
- shadow switch rate: `0.2391`

## Test Once

- test all improvement: `0.0321`
- test t50 improvement: `-0.0083`
- test hard/failure improvement: `0.0655`
- test easy degradation: `0.0527`
- test switch rate: `0.1919`

## Interpretation

- Stage43-CC uses validation-only calibration/holdout to avoid selecting guard parameters on test.
- It tests whether source-family/domain/horizon support can repair the Stage43-CB easy-safety transfer failure.
- Future waypoint labels are train/eval labels only. Guard inputs are predicted risk and model-vs-floor disagreement plus metadata available at inference.
- Deployment remains unchanged unless test easy preservation and lift both pass.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.

## Gate

| gate | passed |
| --- | --- |
| `stage43_cb_precondition_seen` | `True` |
| `fresh_shadow_replay_completed` | `True` |
| `train_only_heads_refit` | `True` |
| `validation_split_internal_only` | `True` |
| `no_test_threshold_tuning` | `True` |
| `inference_safe_guard_features_only` | `True` |
| `shadow_holdout_easy_safe` | `True` |
| `test_easy_preserved` | `False` |
| `test_lift_vs_floor` | `True` |
| `source_family_support_reported` | `True` |
| `domain_horizon_source_breakdown_reported` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
