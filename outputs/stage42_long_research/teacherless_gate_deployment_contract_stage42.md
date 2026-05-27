# Stage42-HF Teacherless Gate Deployment Contract

- source: `fresh_stage42_hf_teacherless_gate_deployment_contract`
- generated_at_utc: `2026-05-27T17:05:05.554299+00:00`
- git_commit: `4385c56`
- input_hash: `dd83c3b3893f6e21ff24053874d9936e1d255f3a6b476bc925da69a7b8ab6c9a`
- gate: `15 / 15`
- verdict: `stage42_hf_teacherless_gate_deployment_contract_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-HF 是 deployment / paper claim contract refresh，不训练、不调 threshold、不下载、不转换。
- Stage42-HE 支持 teacherless proximity-guard switch gate，但仍要求 causal floor fallback。
- teacher gate removal 只限 repaired proximity-guard switch policy；不是 global causal floor removal。
- future endpoint / future waypoint 只可作为 supervised/evaluation labels，不可作为 inference input。
- 不使用 central velocity，不使用 test endpoints 建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Teacherless Gate Evidence

- policy_family: `harm_predictor_gate`
- min_sep: `0.05`
- teacher_gate_used: `False`
- causal_floor_fallback_used: `True`
- global_floor_removal_allowed: `False`
- all improvement: `20.74%`
- t50 improvement: `13.82%`
- t100 raw-frame diagnostic improvement: `13.68%`
- hard/failure improvement: `19.99%`
- easy degradation: `0.00%`
- bootstrap_n: `2000`
- robust_positive_domains: `['ETH_UCY', 'TrajNet', 'UCY']`
- weak_domain_horizon_slices: `[]`

## Contract Decisions

| request | allowed | status | role | denied reasons | required conditions |
| --- | ---: | --- | --- | --- | --- |
| `teacherless_proximity_guarded_switch_gate` | True | `allowed_protected` | `teacherless_proximity_guarded_switch_gate_with_causal_floor_fallback` |  | use the Stage42-HE repaired harm_predictor_gate plus validation-selected proximity guard<br>keep causal floor fallback active<br>report dataset-local/raw-frame 2.5D only<br>do not tune thresholds on test |
| `teacher_gate_removal_for_repaired_gate` | True | `allowed_policy_specific_not_global` | `teacher_gate_removed_only_for_repaired_floor_free_switch_gate` |  | scope the claim to the repaired proximity-guard switch gate<br>state that teacher/floor mechanisms remain required elsewhere<br>keep causal baseline floor fallback and proximity safety guard |
| `causal_floor_removal` | False | `blocked_required_safety_floor` | `forbidden` | Stage42-HE robust candidate still uses causal floor fallback<br>Stage42-HC found zero globally deployable floor-free candidates<br>global floor removal would overclaim beyond protected switch-gate evidence |  |
| `ungated_neural_or_floor_free_global_deployment` | False | `blocked_unsafe` | `forbidden` | ungated/floor-free global policies are not deployable<br>positive raw floor-free switches were unsafe under near-collision/proximity stress<br>protected deployment must preserve fallback and guard |  |
| `partial_t50_floor_relaxation` | True | `allowed_slice_only` | `bounded_validation_backed_t50_slice_relaxation` |  | only use validation-backed mapped t50 slices<br>do not present as global floor-free deployment<br>keep fallback outside validated slices |
| `teacherless_gate_as_paper_claim` | True | `allowed_with_boundary` | `paper_claim_teacherless_switch_gate_evidence` |  | write 'teacherless proximity-guarded switch gate' rather than 'floor-free world model'<br>include causal floor fallback as a required safety mechanism<br>include raw-frame/dataset-local/2.5D limitations<br>state that Stage5C and SMC were not executed |
| `metric_seconds_true3d_foundation_claim` | False | `forbidden` | `forbidden` | dataset-local/raw-frame evidence does not verify metric coordinates or seconds-level horizons<br>current system is not true 3D and not a large-scale foundation world model |  |
| `stage5c_execution_or_smc_enabled` | False | `forbidden` | `forbidden` | Stage5C latent generative execution remains explicitly disabled<br>SMC remains explicitly disabled<br>this contract is a guard/report refresh, not a stochastic generative rollout stage |  |
| `unknown_future_policy_request` | False | `unknown_request_blocked_by_default` | `forbidden` | unknown request is blocked until added to the contract |  |

## Deployment Defaults

- deployable_default: `Stage37/Stage42 protected causal-floor fallback policy`
- teacherless_candidate: `Stage42-HE proximity-guarded harm_predictor_gate`
- required_safety_floor: `causal floor fallback remains required`
- global_floor_free: `blocked`
- ungated_neural: `blocked`
- metric_seconds_true3d_foundation_claim: `forbidden`
- stage5c: `not executed`
- smc: `disabled`

## Gate

| gate | pass |
| --- | ---: |
| `he_input_passed` | True |
| `hd_input_passed` | True |
| `hc_input_passed` | True |
| `teacherless_gate_allowed_protected` | True |
| `teacher_gate_removal_policy_specific` | True |
| `causal_floor_removal_blocked` | True |
| `ungated_global_floor_free_blocked` | True |
| `partial_t50_relaxation_slice_only` | True |
| `paper_claim_bounded` | True |
| `metric_seconds_true3d_foundation_blocked` | True |
| `stage5c_smc_blocked` | True |
| `unknown_requests_default_deny` | True |
| `no_future_or_test_leakage_claim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
