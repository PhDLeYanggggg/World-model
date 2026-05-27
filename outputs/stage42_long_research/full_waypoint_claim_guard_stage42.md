# Stage42-GZ Full-Waypoint Claim Guard

- source: `fresh_stage42_gz_full_waypoint_claim_guard`
- generated_at_utc: `2026-05-27T15:34:47.493823+00:00`
- git_commit: `20906e9`
- input_hash: `ab6da4087569ee454184400617b678a03984c434f47f951089fa272ca2ec15de`
- gate: `18 / 18`
- verdict: `stage42_gz_full_waypoint_claim_guard_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-GZ 是 full-waypoint claim guard，不重新训练，不调 threshold，不执行 Stage5C，不启用 SMC。
- full-waypoint future labels 只允许作为 supervised/eval label，不允许作为 inference input。
- endpoint-only 或 endpoint-linear bridge 成功不能直接写成 learned full-waypoint dynamics 成功。
- ungated full-waypoint neural 仍不可部署。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。

## Claim Rows

| id | status | allowed main claim | claim | evidence | required boundary |
| --- | --- | ---: | --- | --- | --- |
| `GZ-C1` | `allowed_with_boundary` | True | Protected full-waypoint sequence evidence exists and can be cited as protected raw-frame 2.5D world-state evidence. | CM protected full-waypoint all/t50/t100raw/hard = 18.58% / 14.80% / 22.86% / 19.52%; GJ protected_full_waypoint_runtime_supported=True | protected dataset-local/raw-frame 2.5D; not ungated, not metric, not seconds-level |
| `GZ-C2` | `rejected` | False | Endpoint-only or endpoint-linear bridge success is equivalent to learned full-waypoint dynamics. | CM endpoint status=diagnostic_only_not_full_waypoint; linear bridge all=21.03%, protected full-waypoint all=18.58%; endpoint bridge may remain a floor but cannot be counted as learned shape. | endpoint bridge and full-waypoint shape must be reported separately |
| `GZ-C3` | `rejected_by_safety` | False | Ungated full-waypoint neural dynamics is deployable. | CM ungated full-waypoint easy degradation=124.59%; GJ ungated_full_waypoint_deployable=False | deployment requires protected switch / teacher floor |
| `GZ-C4` | `allowed_with_safety_caveat` | True | Common-validation full-waypoint composer has positive protected endpoint-bridge replacement evidence. | CO vs endpoint ADE all/t50/t100raw/hard/easy = 3.02% / 1.50% / 6.12% / 3.28% / 0.25%; use_full_rate=21.35% | protected common-validation composer, not global floor-free replacement |
| `GZ-C5` | `rejected_by_joint_safety` | False | Proximity-aware guard can be omitted without changing the safety claim. | CQ guarded vs endpoint ADE all/t50/hard = 1.77% / 1.07% / 1.93%; near@0.05 delta vs endpoint=-0.06% | safety-sensitive reports must use proximity-aware guard or explicitly mark caveat |
| `GZ-C6` | `allowed_limited_claim` | True | Graph/group consistency can be cited as a protected full-waypoint module, but not as independent neighbor/interaction dominance. | CM graph/group all/t50/hard=22.24% / 15.09% / 22.41%; supported_modules=['history', 'domain_expert', 'safe_switch', 'teacher_floor', 'group_consistency_full_waypoint', 'full_waypoint_shape', 'endpoint_bridge']; blocked_modules=['scene_goal', 'neighbor_interaction', 'JEPA', 'Transformer'] | claim must be group-consistency full-waypoint under protected policy; neighbor_interaction remains blocked as independent main claim |
| `GZ-C7` | `allowed_with_horizon_limit` | True | Unified row-level full-waypoint evidence supports broad protected source/domain evidence, but not uniform horizon success. | W unified ADE all/t50/t100raw/hard/easy mean = 9.93% / 9.40% / 8.48% / 10.49% / 0.24% | source/domain protected evidence only; weak horizon/h100 blocker remains separate |
| `GZ-C8` | `rejected_by_protocol_boundary` | False | Global primary full-waypoint replacement claim is allowed. | GJ global_primary_full_waypoint_replacement_claim_allowed=False; reason=Fresh DI/DL support a protected source-level group-consistency full-waypoint runtime policy with exact replay and proximity repair. However, common-validation endpoint-linear bridge/composer and source-level train-horizon floor use different comparison protocols, so the result cannot be collapsed into a single global primary full-waypoint replacement claim. | report source-level/protected components; do not collapse protocols into global primary replacement |
| `GZ-C9` | `rejected_global_boundary` | False | Metric/seconds-level, true-3D, foundation, Stage5C, or SMC claims are allowed. | All inputs are dataset-local/raw-frame 2.5D evidence; Stage5C and SMC remain false. | raw-frame / dataset-local only |

## Deployment Interpretation

- Protected full-waypoint evidence can be cited only as dataset-local/raw-frame 2.5D world-state evidence.
- Endpoint-only and endpoint-linear bridge evidence remain separate from learned full-waypoint shape evidence.
- Ungated full-waypoint neural deployment remains rejected.
- Group-consistency full-waypoint is a supported protected module; neighbor/interaction alone remains blocked as an independent main claim.
- Uniform horizon / h100 success is not claimed here.
