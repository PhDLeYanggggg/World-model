# Stage43-CB Downstream Easy Guard Audit

- source: `fresh_stage43_cb_downstream_easy_guard_audit`
- result_source: `fresh_validation_only_easy_guard_replay`
- verdict: `stage43_cb_downstream_easy_guard_val_safe_test_easy_mismatch`
- gate: `12 / 13`
- selected latent variant: `identity_stage43m_adapter_z`
- evaluated validation policies: `16464`
- validation-safe policies: `16292`
- deployable policy changed: `False`

## Validation-Selected Policy

- policy: `{'gain_threshold': 0.85, 'harm_threshold': 0.1, 'failure_threshold': 0.5, 'disagreement_threshold': 2.0910258293151855, 'endpoint_disagreement_threshold': 0.6212336719036102}`
- validation all improvement: `0.1313`
- validation t50 improvement: `-0.0057`
- validation hard/failure improvement: `0.1688`
- validation easy degradation: `0.0002`
- validation switch rate: `0.2416`

## Test Once

- test all improvement: `0.0321`
- test t50 improvement: `-0.0083`
- test hard/failure improvement: `0.0656`
- test easy degradation: `0.0528`
- test switch rate: `0.1919`

## Validation-Test Gap

- easy degradation gap: `0.0526`
- all improvement gap: `-0.0992`
- t50 improvement gap: `-0.0025`

## Interpretation

- Stage43-CB refits the downstream heads on train only, then searches a stricter easy guard on validation only.
- The guard uses predicted failure/gain/harm plus model-vs-floor rollout disagreement. These are inference-safe quantities.
- The test set is evaluated once with the validation-selected policy.
- A validation-safe policy can still harm test easy rows. That is a deployment blocker, not a success.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.

## Gate

| gate | passed |
| --- | --- |
| `stage43_ca_precondition_seen` | `True` |
| `fresh_guard_replay_completed` | `True` |
| `train_only_heads_refit` | `True` |
| `future_labels_eval_only` | `True` |
| `no_test_threshold_tuning` | `True` |
| `inference_safe_guard_features_only` | `True` |
| `validation_easy_safe_policy_found` | `True` |
| `test_easy_preserved` | `False` |
| `protected_lift_vs_floor` | `True` |
| `validation_test_easy_gap_reported` | `True` |
| `domain_horizon_source_breakdown_reported` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
