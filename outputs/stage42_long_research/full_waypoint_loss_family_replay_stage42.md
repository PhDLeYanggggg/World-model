# Stage42-DX Full-Waypoint Loss-Family Fresh Replay

- source: `fresh_rerun_dg_dh_loss_family_replay`
- generated_at_utc: `2026-05-27T00:52:21.732532+00:00`
- gate: `10 / 10`
- verdict: `stage42_dx_loss_family_replay_pass_blocker_confirmed`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-DX fresh-reruns Stage42-DG and Stage42-DH full-waypoint loss-family probes, then applies one promotion gate.
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- candidate_count: `2`
- any_promotable_over_stage42_am: `False`
- promotion_decision: `do_not_promote_keep_stage42_am_or_cq_floor`
- promotion_blockers: `['no_loss_family_candidate_beats_stage42_am_on_all_and_hard', 'primary_full_waypoint_promotion_blocked', 'next_step_requires_model_architecture_or_explicit_physical_consistency_target_not_more_scalar_weighting']`
- best_candidate_name: `proximity_occupancy_loss`
- best_candidate_all: `0.25506106152262753`
- best_candidate_t50: `0.22136644201311806`
- best_candidate_hard: `0.2373927398811264`
- best_candidate_easy: `-0.29229256390643454`

## Candidate Replay Table

| candidate | selected | all | t50 | t100 raw | hard | easy | delta all vs AM | delta hard vs AM | promotable |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `all_hard_weighted_loss` | `balanced:100.0` | 0.245788 | 0.220171 | 0.143652 | 0.237494 | -0.256627 | 0.000000 | 0.000000 | `False` |
| `proximity_occupancy_loss` | `proximity_close_weighted:100.0` | 0.255061 | 0.221366 | 0.143395 | 0.237393 | -0.292293 | 0.009273 | -0.000101 | `False` |

## Interpretation

- This is a fresh replay of the two strongest full-waypoint loss-family repair tracks, not a threshold-only report.
- Promotion requires beating Stage42-AM on both all and hard/failure while keeping easy degradation <=2%.
- If no candidate is promotable, the current deployable path remains Stage42-AM/CQ/CS guarded floor, not primary full-waypoint dynamics.

## Gate

| gate | pass |
| --- | --- |
| `dg_fresh_rerun_completed` | `True` |
| `dh_fresh_rerun_completed` | `True` |
| `loss_family_candidates_compared` | `True` |
| `validation_selected_candidates` | `True` |
| `no_future_or_test_leakage` | `True` |
| `honest_promotion_decision_recorded` | `True` |
| `promotion_blocker_recorded_if_needed` | `True` |
| `no_metric_seconds_overclaim` | `True` |
| `stage5c_false` | `True` |
| `smc_false` | `True` |
