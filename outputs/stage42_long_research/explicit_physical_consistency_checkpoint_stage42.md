# Stage42-DY Explicit Physical Consistency Checkpoint

- source: `fresh_dg_dh_di_physical_consistency_checkpoint`
- generated_at_utc: `2026-05-27T01:07:48.809280+00:00`
- gate: `16 / 16`
- verdict: `stage42_dy_explicit_physical_consistency_checkpoint_pass_source_level_promoted`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-DY 是 Stage42-DX 后的显式 physical/group-consistency checkpoint，不继续重复 scalar loss weighting。
- Stage42-DY fresh-runs DG/DH loss-family probes and DI group-consistency repair in one comparison frame.
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Decision Summary

- loss-family any promotable over Stage42-AM: `False`
- best loss-family candidate: `proximity_occupancy_loss`
- best loss-family all/t50/hard/easy: `0.255061` / `0.221366` / `0.237393` / `-0.292293`
- group-consistency promotable over Stage42-AM: `True`
- group-consistency all/t50/t100 raw/hard/easy: `0.247157` / `0.223630` / `0.143461` / `0.238874` / `-0.256309`
- group-consistency delta vs Stage42-AM all/hard: `0.001368` / `0.001380`
- group-consistency delta vs best loss-family all/hard: `-0.007904` / `0.001481`
- near@0.05 base/final: `0.019364` / `0.013823`
- deployment_decision: `promote_explicit_group_consistency_as_source_level_full_waypoint_physical_policy`

## Interpretation

- Stage42-DX confirmed scalar loss-family weighting is not enough: no loss-family candidate beats Stage42-AM on both all and hard/failure.
- Stage42-DY confirms the next useful route is explicit physical/group consistency over predicted all-agent full-waypoint rollouts.
- The group-consistency policy beats Stage42-AM on all and hard/failure and repairs near-collision, but it is source-level train-horizon-floor evidence, not a global primary full-waypoint replacement.
- The best scalar loss candidate still has slightly better all-ADE than group consistency, so the correct claim is not 'one universal winner'; the correct claim is a protocol-bounded source-level physical consistency policy plus guarded bridge/composer policies.

## Gate

| gate | pass |
| --- | --- |
| `dg_loss_probe_fresh` | `True` |
| `dh_loss_probe_fresh` | `True` |
| `di_group_consistency_fresh` | `True` |
| `loss_family_blocker_confirmed` | `True` |
| `group_consistency_promotable` | `True` |
| `group_consistency_beats_am_all` | `True` |
| `group_consistency_beats_am_hard` | `True` |
| `group_consistency_hard_not_worse_than_best_loss` | `True` |
| `loss_family_all_advantage_recorded` | `True` |
| `near_collision_repaired` | `True` |
| `no_future_or_test_leakage` | `True` |
| `global_primary_overclaim_blocked` | `True` |
| `ungated_full_waypoint_blocked` | `True` |
| `no_metric_seconds_overclaim` | `True` |
| `stage5c_false` | `True` |
| `smc_false` | `True` |
