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
- Stage41 verdict: `composite_tail_safe_switch_bounded_neural_dynamics_candidate`
- gates: `41 / 41`
- best candidate: `composite_tail_safe_switch_bounded_neural_dynamics`
- deployment state: `composite_tail_candidate_pending_final_package_acceptance`
- current strongest neural candidate: `M3W-Neural v1 composite-tail safe-switch bounded neural dynamics under Stage37/teacher safety floor`
- current fallback floor: `Stage37 selector`

## Key Numbers

- all improvement vs Stage37 floor: `21.03%`
- t+50 improvement vs Stage37 floor: `13.65%`
- t+100 raw-frame diagnostic improvement: `14.69%`
- hard/failure improvement: `20.38%`
- easy degradation: `0.00%`
- positive external domains: `3`
- bootstrap evidence pass: `True`
- multiseed replication pass: `True`
- pure UCY source-heldout gate: `True`
- all-agent composite world-state pass: `True`
- all-agent composite ADE all/t+50/t+100: `21.03%` / `13.65%` / `14.69%`
- all-agent composite FDE all/t+50: `19.82%` / `17.39%`
- strict pure UCY-only retrain/select/test gate: `False`
- strict pure UCY neural retrain gate: `True`
- strict pure UCY neural best trial/mode: `pure_ucy_transformer` / `bounded_endpoint_residual`
- strict pure UCY neural best metrics all/t+50/hard/easy: `9.01%` / `8.80%` / `9.36%` / `0.00%`
- strict pure UCY neural blocker: ``
- endpoint-to-full bridge gate: `True`
- endpoint-to-full bridge positive domains: `['ETH_UCY', 'TrajNet']`
- calibrated learned-shape meta-policy gate: `True`
- calibrated learned-shape positive domains: `['ETH_UCY', 'TrajNet']`
- JEPA deployable path: `disable_jepa_in_deployable_path`
- fixed-prior source switch beats fixed composer: `False`
- residual source-switch oracle headroom: `False`

## Safety

- endpoint geometry pass: `True`
- no leakage: `{'future_endpoint_input': False, 'future_endpoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False}`
- future endpoint is label/eval only.
- deployment remains gated under Stage37/teacher safety floor; raw full-row neural blends and ungated endpoint dynamics are not claimed safe.

## What This Does Not Claim

- 不是 true 3D。
- 不是 foundation world model。
- 不是 metric prediction。
- 不是 seconds-level horizon。
- 不是 Stage5C latent generative rollout。
- 不是 SMC。

## Current Best Deployable Answer

M3W-Neural v1 composite-tail is the strongest current protected neural dynamics candidate. It has bootstrap, multiseed, pure-UCY source-heldout support, and a full active-agent composite waypoint rollout audit. It remains a protected candidate, not an ungated neural replacement. The stricter pure UCY-only neural retrain/select/test audit has now been attempted and failed deployability because source-shift/easy-safety was not reliable, so Stage37 remains the explicit safety floor. A new endpoint-to-full bridge audit is positive on ETH_UCY and TrajNet, showing endpoint neural dynamics can survive actual full-waypoint evaluation through a linear bridge. The calibrated learned-shape meta-policy then adds small but positive protected waypoint-shape residual contribution on both domains.

Recent negative source-switch and strict pure-UCY neural retrain audits show that residual source selection and source-only neural retraining are not the next useful deployment path without new causal features, stronger scene/domain context, or more independent UCY-like validation data.
