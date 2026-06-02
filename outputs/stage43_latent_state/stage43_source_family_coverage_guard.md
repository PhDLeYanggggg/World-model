# Stage43-CD Source-Family Coverage Guard

- source: `fresh_stage43_cd_source_family_coverage_guard`
- result_source: `fresh_validation_source_family_coverage_guard`
- verdict: `stage43_cd_source_family_coverage_guard_pass`
- gate: `14 / 14`
- selected policy: `domain_source_family_coverage_guard`
- deployable policy changed: `False`

## Coverage Evidence

- selection rule: Select a shadow-holdout-safe policy by validation objective; if multiple policies tie within 1e-6, prefer the stricter source-family coverage guard. No test rows or test metrics are used.
- validation source families: `['biwi', 'hotel', 'students']`
- test global-unsupported families: `{'pets': 1926, 'zara': 9540}`
- test domain-unsupported families: `{'ETH_UCY|students': 70585, 'TrajNet|pets': 1926, 'UCY|zara': 9540}`

## Shadow Holdout

- all improvement: `0.1321`
- t50 improvement: `-0.0079`
- hard/failure improvement: `0.1690`
- easy degradation: `0.0010`
- switch rate: `0.2391`

## Test Once

- all improvement: `0.0111`
- t50 improvement: `-0.0000`
- hard/failure improvement: `0.0135`
- easy degradation: `0.0000`
- switch rate: `0.0227`

## Interpretation

- Stage43-CD repairs the Stage43-CB/CC easy-safety mismatch by refusing learned switches on source families not covered by validation.
- This is a source-coverage safety protocol, not a new test-tuned threshold.
- It preserves easy safety and keeps a small all-row lift, but it sacrifices some hard/failure lift and still does not repair t50.
- Deployment remains unchanged until this guard is reconciled with the current frozen deployable policy family.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.

## Gate

| gate | passed |
| --- | --- |
| `stage43_cc_precondition_seen` | `True` |
| `fresh_coverage_replay_completed` | `True` |
| `train_only_heads_refit` | `True` |
| `validation_shadow_only_selection` | `True` |
| `coverage_guard_selected` | `True` |
| `unsupported_test_families_reported` | `True` |
| `shadow_holdout_easy_safe` | `True` |
| `test_easy_preserved` | `True` |
| `test_lift_vs_floor` | `True` |
| `test_t50_reported` | `True` |
| `test_hard_failure_reported` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
