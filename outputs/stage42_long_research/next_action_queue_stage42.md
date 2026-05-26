# Stage42-DA Next-Action Evidence Queue

- source: `fresh_synthesis_from_cached_verified_stage42_artifacts`
- generated_at_utc: `2026-05-26T20:49:23.161383+00:00`
- git_commit: `05c987f`
- gate: `15 / 15`
- verdict: `stage42_da_next_action_queue_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-DA 是 next-action evidence queue，不重新训练，不调 threshold，不把计划当完成。
- 所有下一步动作必须继续区分 fresh_run / cached_verified / not_run。
- future endpoints / future waypoints 只能作为 supervised/evaluation labels，不能作为 inference input。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Current Evidence Snapshot

- paper_freeze_status: `candidate_clean`
- paper_freeze_final_immutable_release: `True`
- stage42_dirty_files: `0`
- stage42_substantive_dirty_files: `0`
- dominant_mechanism: `baseline_family_rollout_context`
- goal_scene_rescue_success: `False`
- neighbor_interaction_rescue_success: `False`
- common_validation_composer_all_improvement: `0.030166976195252437`
- proximity_guard_all_improvement: `0.017743597342181783`
- proximity_guard_t50_improvement: `0.010673426149055754`
- global_t100_claim_ready: `False`

## Evidence Files Checked

- paper_freeze: `True`
- worktree_caveat: `True`
- a_journal_gap: `True`
- context_forensics: `True`
- goal_scene: `True`
- neighbor_interaction: `True`
- bridge_shape: `True`
- proximity_guard: `True`
- proximity_ablation: `True`
- source_terms: `True`
- source_time_geometry: `True`
- t100_gap: `True`

## Prioritized Next Actions

### DA-1 - Close legal/source support for ETH_UCY and TrajNet t100/t50 calibration

- priority: `100`
- status: `not_run_next_action`
- why_now: Global t100 and restricted metric/seconds claims are still blocked by independent source support and terms confirmation.
- requires_user_or_external_state: `True`
- blocked_claim_until_done: `global_or_restricted metric/seconds and global t100 deployable claim`
- success_gate: official/terms-safe conversion + no-leakage + source-CV positive/easy-safe on ETH_UCY and TrajNet.
- evidence:
  - `outputs/stage42_long_research/t100_data_gap_audit_stage42.md`
  - `outputs/stage42_long_research/source_terms_validation_stage42.md`
  - `outputs/stage42_long_research/source_time_geometry_calibration_stage42.md`
- next_commands:
  - `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`
  - `.venv-pytorch/bin/python run_stage42_source_time_geometry_calibration.py`

### DA-2 - Train a stronger source-compatible sequence/graph context model beyond baseline-family rollout

- priority: `92`
- status: `not_run_next_action`
- why_now: Current context forensics says baseline-family rollout context dominates; history/goal/neighbor/interaction are not independent main drivers yet.
- requires_user_or_external_state: `False`
- blocked_claim_until_done: `scene/goal/neighbor/interaction as independent main contribution`
- success_gate: retrained context model beats baseline-family control on all/t50/hard with easy degradation <=2%.
- evidence:
  - `outputs/stage42_long_research/context_contribution_forensics_stage42.md`
  - `outputs/stage42_long_research/goal_scene_gated_expert_stage42.md`
  - `outputs/stage42_long_research/neighbor_interaction_gated_expert_stage42.md`
- next_commands:
  - `.venv-pytorch/bin/python run_stage42_source_level_sequence_context.py`
  - `.venv-pytorch/bin/python run_stage42_source_level_graph_context.py`

### DA-3 - Promote protected full-waypoint from bridge/composer to learned all-agent sequence dynamics

- priority: `88`
- status: `not_run_next_action`
- why_now: Common-validation composer is positive, but endpoint-linear bridge remains the stronger all-ADE floor and ungated full-waypoint is unsafe.
- requires_user_or_external_state: `False`
- blocked_claim_until_done: `ungated or independently learned full-waypoint world dynamics`
- success_gate: learned sequence dynamics improves endpoint-linear/proximity-guard composer on all/t50/hard without proximity/easy regression.
- evidence:
  - `outputs/stage42_long_research/full_waypoint_bridge_shape_audit_stage42.md`
  - `outputs/stage42_long_research/common_validation_bridge_shape_composer_stage42.md`
  - `outputs/stage42_long_research/proximity_guard_ablation_stage42.md`
- next_commands:
  - `.venv-pytorch/bin/python run_stage42_full_waypoint_dynamics.py`
  - `.venv-pytorch/bin/python run_stage42_common_validation_bridge_shape_composer.py`

### DA-4 - Convert paper-freeze candidate into reviewer-replay package

- priority: `76`
- status: `not_run_next_action`
- why_now: Stage42-CZ has a clean hash manifest, but a reviewer still needs a minimal replay sequence and immutable archive boundary.
- requires_user_or_external_state: `False`
- blocked_claim_until_done: `paper-ready reproducibility package beyond internal manifest`
- success_gate: manifest, replay, and provenance can be regenerated without tracked artifact churn.
- evidence:
  - `outputs/stage42_long_research/paper_freeze_candidate_manifest_stage42.md`
  - `outputs/stage42_long_research/evidence_provenance_stage42.md`
  - `outputs/stage42_long_research/proximity_guard_batch_replay_stage42.md`
- next_commands:
  - `.venv-pytorch/bin/python run_stage42_paper_freeze_candidate_manifest.py`
  - `.venv-pytorch/bin/python -m pytest tests/test_stage42_paper_freeze_candidate_manifest.py tests/test_stage42_evidence_provenance_verifier.py`

### DA-5 - Audit deployment variants as safety-sensitive vs accuracy-priority policies

- priority: `72`
- status: `not_run_next_action`
- why_now: No-guard composer has higher ADE, proximity guard is safer. Paper and deployment must not mix these claims.
- requires_user_or_external_state: `False`
- blocked_claim_until_done: `single deployment policy claim with explicit risk/accuracy tradeoff`
- success_gate: deployment card separates safety-sensitive deployable from diagnostic accuracy-priority variant.
- evidence:
  - `outputs/stage42_long_research/proximity_aware_composer_guard_stage42.md`
  - `outputs/stage42_long_research/proximity_guard_ablation_stage42.md`
- next_commands:
  - `.venv-pytorch/bin/python run_stage42_proximity_guard_ablation.py`
  - `.venv-pytorch/bin/python run_stage42_runtime_replay_paper_refresh.py`

## Interpretation

- Stage42-DA does not count any next action as complete.
- It converts current evidence and blockers into an ordered experiment queue.
- The strongest current deployable claim remains the Stage42-CQ/CV proximity-aware guarded composer under Stage37/teacher floor.
- The next substantive research risk is proving independent neural/scene/interaction/full-waypoint contribution beyond baseline-family rollout context.
