# Stage42 Method Draft

## Problem

Given past-only multi-agent history, neighbor context, goal/prototype context, and a strongest causal/Stage37 teacher floor, predict future endpoint and full-waypoint world-state trajectories under strict no-leakage constraints.

## Inputs

- past-only history windows and causal velocities
- neighbor/interaction/group-consistency features
- domain/horizon metadata
- train-only goal/prototype features where available
- Stage37/teacher floor rollout and proposal scores

No future endpoint, future waypoint, central velocity, or test endpoint goal is used as inference input.

## Model

The deployable path is a composite-tail safe-switch bounded neural dynamics policy under the Stage37/teacher floor. It combines a validation-selected teacher repaired switch with a small bounded tail alpha for confident neural proposals. Stage42-C additionally evaluates a protected full-waypoint sequence model on reconstructed future waypoint labels.

## Safety

Stage42-E evaluates internal self-gates, uncertainty gates, harm gates, conformal-style risk gates, teacher-prob gates, and bounded residual blends. The current deployable conclusion is that the Stage37/teacher floor remains necessary. Ungated neural improves raw error but fails safety.

## Claim Boundary

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- SDD 是 pixel-space；external 是 dataset-local / unverified weak metric diagnostic。
- t+50 / t+100 是 raw-frame horizon，不能写成 seconds-level。
- global metric/time claims 仍不允许；TGSIM 只能作为 traffic diagnostic，不是 pedestrian official claim。
- self-audited / visual-prior labels 不是 human gold。
- Stage5C latent generative 未执行。
- SMC 未启用。
