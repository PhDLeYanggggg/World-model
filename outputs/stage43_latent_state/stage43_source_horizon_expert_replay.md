# Stage43-AX Source-Horizon Expert Replay

- source: `fresh_stage43_ax_source_horizon_expert_replay`
- result_source: `fresh_exact_replay_from_stage43_aw_artifact`
- verdict: `stage43_ax_source_horizon_expert_replay_pass`
- gate: `12 / 12`
- reviewer replay passed: `True`
- candidate for deployment update: `True`
- deploy without replay: `False`
- policy hash: `824b4be3be967cc96872fa8c627eb141e9859f2a01eb15d92697507441e38f22`
- row hash: `9d27e3a5fba7583152ed8fb175f21685e989a7be93f23d12c2e4aba36bd1212c`
- switch hash: `57fbceb1da086de79e55905f9e9b537e91162b743b76776e5e50dea6651975c9`
- replay max metric diff: `0.0000000000`

## Replayed Policy

- selected t50 expert: `t50_guard0.03_eth0.15_traj0.10`
- replay mode: artifact selected policy only; no validation reselection and no test threshold tuning

## Replayed Metrics

- all improvement: `23.40%`
- t50 improvement: `12.80%`
- t100 raw-frame diagnostic: `1.35%`
- hard/failure improvement: `24.73%`
- easy degradation: `0.00%`
- switch rate: `19.37%`

## Bootstrap CI

- all CI: `[23.05%, 23.76%]`
- t50 CI: `[12.16%, 13.51%]`
- hard/failure CI: `[24.34%, 25.12%]`
- easy degradation CI: `[0.00%, 0.00%]`

## Metric Replay Diff

| metric | artifact | replayed | abs diff |
| --- | ---: | ---: | ---: |
| `all_improvement_vs_floor` | `23.40%` | `23.40%` | `0.0000000000` |
| `t50_improvement_vs_floor` | `12.80%` | `12.80%` | `0.0000000000` |
| `t100_raw_frame_diagnostic_vs_floor` | `1.35%` | `1.35%` | `0.0000000000` |
| `hard_failure_improvement_vs_floor` | `24.73%` | `24.73%` | `0.0000000000` |
| `easy_degradation_vs_floor` | `0.00%` | `0.00%` | `0.0000000000` |
| `switch_rate` | `19.37%` | `19.37%` | `0.0000000000` |

## Boundary

- This is exact reviewer replay of the Stage43-AW artifact, not a new threshold search.
- Dataset-local/raw-frame 2.5D only.
- No metric/seconds, true 3D, foundation, Stage5C, or SMC claim.

## Gate

| gate | passed |
| --- | --- |
| `stage43_aw_artifact_present` | `True` |
| `artifact_requires_reviewer_replay` | `True` |
| `policy_hash_recorded` | `True` |
| `row_hash_recorded` | `True` |
| `switch_hash_recorded` | `True` |
| `replay_metrics_exact` | `True` |
| `replayed_t50_positive` | `True` |
| `replayed_aggregate_safe` | `True` |
| `domain_easy_safe` | `True` |
| `source_negative_free` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
