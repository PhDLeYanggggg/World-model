# M3W-Neural v1 Frozen Evidence Report

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D multi-agent trajectory world-state model。
- SDD 是 pixel-space benchmark；external 是 dataset-local / unverified weak-metric diagnostic。
- t+50 / t+100 是 raw-frame horizons，不能写成 seconds-level。
- homography / metric scale / effective seconds 未验证。
- self-audited / visual-prior labels 不是 human gold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Verdict

- package result source: `cached_verified`
- Stage41 verdict: `stage41_self_gated_neural_candidate_endpoint_geometry_verified`
- gates: `41 / 41`
- best candidate: `fresh_self_gated_endpoint::binary_fde_neural_dynamics`
- deployment state: `stage41_protected_neural_candidate_pending_user_acceptance`
- current strongest neural candidate: `M3W-Neural v1 self-gated endpoint dynamics under Stage37 safety floor`
- current fallback floor: `Stage37 selector`

## Key Numbers

- all improvement vs Stage37 floor: `41.96%`
- t+50 improvement vs Stage37 floor: `40.62%`
- t+100 raw-frame diagnostic improvement: `45.73%`
- hard/failure improvement: `43.61%`
- easy degradation: `0.00%`
- positive external domains: `3`

## Safety

- endpoint geometry pass: `True`
- no leakage: `{'future_endpoint_input': False, 'future_endpoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False}`
- future endpoint is label/eval only.
- deployment remains gated; raw ungated endpoint dynamics are not claimed safe.

## What This Does Not Claim

- 不是 true 3D。
- 不是 foundation world model。
- 不是 metric prediction。
- 不是 seconds-level horizon。
- 不是 Stage5C latent generative rollout。
- 不是 SMC。

## Current Best Deployable Answer

M3W-Neural v1 is frozen as the first Stage41 gate-passing protected neural candidate. It should be treated as a candidate pending user acceptance and broader protocol replication; Stage37 remains the explicit safety floor.
