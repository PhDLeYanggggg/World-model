# Stage42-FY Horizon Retry Decision Map

- source: `fresh_stage42_horizon_retry_decision_map_from_fl_fq`
- generated_at_utc: `2026-05-27T10:55:08.523883+00:00`
- git_commit: `c1a3d27`
- input_hash: `33c730670dff7f2f643c638bbc9d250f3d1af0f326cf5d469a646e18b93e1ba6`
- gate: `14 / 14`
- verdict: `stage42_fy_horizon_retry_decision_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-FY does not train, evaluate, download, convert, or tune thresholds.
- It decides when weak-horizon modeling retries are scientifically justified.
- Stage5C is not executed; SMC is not enabled.

## Summary

- weak_horizons: `['TrajNet|100', 'UCY|100']`
- model_retry_attempts_considered: `5`
- promoted_policy_count: `0`
- stop_repeat_modeling_now: `True`
- uniform_horizon_claim_allowed: `False`
- highest_priority_data_action: `FW-TERMS-ucy_crowd_original`

## Prior Attempts Considered

| attempt | verdict | outcome | policy promoted |
| --- | --- | --- | ---: |
| `FL weak-horizon forensics` | `stage42_fl_horizon_weak_slice_forensics_pass` | diagnosed low-margin ambiguous oracle labels | False |
| `FM row-level switch specialist` | `stage42_fm_horizon_row_switch_specialist_pass_with_horizon_limit` | repaired one weak horizon but left TrajNet|100 and UCY|100 weak | False |
| `FN conservative easy guard` | `stage42_fn_conservative_easy_guard_pass_with_horizon_limit` | did not repair remaining h100 weak horizons | False |
| `FO gain/harm specialist` | `stage42_fo_gain_harm_specialist_pass_with_horizon_limit` | partial global metrics but uniform horizon claim still blocked | False |
| `FP/FQ source-support audit and repair queue` | `stage42_fq_h100_source_support_repair_queue_pass` | TrajNet h100 needs longer legal source; UCY h100 needs terms and guarded conversion | False |

## Retry Decisions

| weak key | decision | required before retry | blocked retries | next actions |
| --- | --- | --- | --- | --- |
| `TrajNet|100` | `stop_model_retry_until_longer_legal_source` | official longer TrajNet-compatible raw source<br>terms confirmation<br>guarded conversion<br>no-leakage audit<br>train-only source-CV | repeat row_switch on same features<br>repeat conservative easy guard on same features<br>repeat gain/harm specialist on same source support<br>claim uniform horizon robustness | FW-H100-TrajNet|100<br>FW-DOMAIN-TrajNet<br>FW-TERMS-trajnetplusplus_official |
| `UCY|100` | `stop_model_retry_until_terms_and_guarded_conversion` | UCY original terms/user confirmation<br>guarded conversion of local h100 candidates<br>no-leakage audit<br>train-only source-CV | repeat row_switch on same features<br>repeat conservative easy guard on same features<br>repeat gain/harm specialist on same source support<br>claim uniform horizon robustness | FW-TERMS-ucy_crowd_original<br>FW-H100-UCY|100<br>FW-DOMAIN-UCY |

## Gate

| gate | pass |
| --- | ---: |
| `source_fresh` | True |
| `weak_horizons_identified` | True |
| `multiple_model_retries_considered` | True |
| `no_policy_promoted_from_failed_retries` | True |
| `stop_repeat_modeling_now` | True |
| `uniform_horizon_not_claimed` | True |
| `every_weak_key_has_decision` | True |
| `each_decision_has_allowed_retry_conditions` | True |
| `each_decision_has_user_actions` | True |
| `no_download_conversion_training_eval` | True |
| `no_metric_seconds_overclaim` | True |
| `no_true3d_foundation_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |

## Interpretation

- Current weak h100 horizons should not receive more same-feature model retries until new legal source support or train-only validation support exists.
- This protects the research loop from overfitting or repeatedly tuning on weak, low-margin slices.
- The correct next move is source/legal support closure, not claiming uniform horizon robustness.
