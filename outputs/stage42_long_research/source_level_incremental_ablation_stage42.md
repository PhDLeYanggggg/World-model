# Stage42-AO Proposed Source-Level Incremental Ablation

- source: `fresh_run`
- generated_at_utc: `2026-05-29T05:21:02.598251+00:00`
- git_commit: `5a83e6d`
- input_hash: `9df575dc442002dae50e94edcffb64e4435fe18c35c72bf2ad673c885accea51`
- gate: `10 / 11`
- verdict: `stage42_ao_incremental_component_evidence_partial_or_negative`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-AO 是 proposed source-level split incremental / standalone retrained ablation，不是 metric 或 seconds-level 结果。
- 每个 variant 都重新训练 ridge full-waypoint probe，并在 validation 上重新选 safe policy；test 只评一次。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Why This Was Run

- Stage42-AN showed that full-minus-module retrained ablation only clearly supported `baseline_family_context`.
- Stage42-AO asks a sharper question: do history / goal / neighbor context features have standalone signal, or incremental value after baseline-family rollout context?
- A negative result is still useful: it tells us not to write these modules as independent paper claims under the current ridge source-level feature set.

## Variant Metrics

| variant | features | all | t50 | t100 diag | hard/failure | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | 166 | 0.245788 | 0.220171 | 0.143652 | 0.237494 | -0.256627 | 0.588099 |
| `horizon_domain_only` | 7 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 |
| `baseline_family_only` | 35 | 0.287773 | 0.315425 | 0.142825 | 0.275812 | -0.324186 | 0.660626 |
| `history_only` | 128 | 0.175400 | 0.199761 | -0.002730 | 0.168080 | -0.141575 | 0.469341 |
| `goal_only` | 17 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 |
| `neighbor_only` | 12 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 |
| `motion_goal_context` | 138 | 0.178810 | 0.202283 | 0.000074 | 0.171817 | -0.151570 | 0.474757 |
| `baseline_plus_history` | 156 | 0.258387 | 0.226508 | 0.143817 | 0.240185 | -0.313018 | 0.616651 |
| `baseline_plus_goal` | 45 | 0.262502 | 0.227640 | 0.142748 | 0.248648 | -0.301712 | 0.633739 |
| `baseline_plus_neighbor` | 40 | 0.263747 | 0.229616 | 0.143182 | 0.248791 | -0.313188 | 0.628977 |
| `baseline_plus_history_goal_neighbor` | 166 | 0.245788 | 0.220171 | 0.143652 | 0.237494 | -0.256627 | 0.588099 |

## Incremental Evidence

| variant | delta all vs baseline-family | delta t50 | delta hard/failure | standalone positive | incremental positive | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `history_only` | -0.112373 | -0.115665 | -0.107732 | True | False | history_only has standalone positive signal versus the train-horizon floor while preserving easy cases. |
| `goal_only` | -0.287773 | -0.315425 | -0.275812 | False | False | goal_only does not show standalone positive signal strong enough for a paper module claim. |
| `neighbor_only` | -0.287773 | -0.315425 | -0.275812 | False | False | neighbor_only does not show standalone positive signal strong enough for a paper module claim. |
| `motion_goal_context` | -0.108963 | -0.113143 | -0.103995 | True | False | motion_goal_context has standalone positive signal versus the train-horizon floor while preserving easy cases. |
| `baseline_plus_history` | -0.029387 | -0.088917 | -0.035627 | True | False | baseline_plus_history does not improve over baseline_family_only by > 0.01; baseline-family context likely absorbs this signal here. |
| `baseline_plus_goal` | -0.025271 | -0.087785 | -0.027164 | True | False | baseline_plus_goal does not improve over baseline_family_only by > 0.01; baseline-family context likely absorbs this signal here. |
| `baseline_plus_neighbor` | -0.024026 | -0.085810 | -0.027021 | True | False | baseline_plus_neighbor does not improve over baseline_family_only by > 0.01; baseline-family context likely absorbs this signal here. |
| `baseline_plus_history_goal_neighbor` | -0.041985 | -0.095255 | -0.038318 | True | False | baseline_plus_history_goal_neighbor does not improve over baseline_family_only by > 0.01; baseline-family context likely absorbs this signal here. |

## Summary

- positive_standalone_context_variants: `['history_only', 'motion_goal_context']`
- positive_incremental_context_variants: `[]`
- component_evidence_verdict: `stage42_ao_context_components_not_independently_supported`
- full_minus_baseline_family_only: `{'all_improvement': -0.041985018166175725, 't50_improvement': -0.09525464052523891, 't100_raw_frame_diagnostic_improvement': 0.000827649151606602, 'hard_failure_improvement': -0.03831835955182494, 'easy_degradation': 0.06755852549095309, 'switch_rate': -0.07252728728559987, 'harm_over_fallback': 0.01933577492187434}`

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'family_fde_input': False, 'safe_strongest_idx_old_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`

## Verification

- focused pytest: `.venv-pytorch/bin/python -m pytest tests/test_stage42_source_level_incremental_ablation.py tests/test_stage42_current_module_claim_refresh.py -> 7 passed in 0.76s`
- full pytest: `.venv-pytorch/bin/python -m pytest tests -> 1196 passed in 827.27s (0:13:47)`

## Interpretation

- Stage42-AO did not find enough standalone/incremental context-module evidence under this ridge protocol; the current evidence remains dominated by baseline-family rollout context.
- This does not prove history/goal/neighbor are useless; it means their independent paper claim requires a stronger neural/graph retraining protocol or richer source-level context.
- Claims remain dataset-local raw-frame 2.5D, not metric/seconds-level, true-3D, foundation, Stage5C, or SMC evidence.
