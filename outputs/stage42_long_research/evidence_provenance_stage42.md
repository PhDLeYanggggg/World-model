# Stage42-CX Evidence Provenance Verifier

- source: `fresh_evidence_provenance_from_stage42_artifacts`
- generated_at_utc: `2026-05-26T22:57:58.794488+00:00`
- git_commit: `31d01c1`
- input_hash: `36005ac6fce16f6b09d201807b285be3d0080a91c7335ece1beb7e028b15de73`
- gate: `20 / 20`
- verdict: `stage42_cx_evidence_provenance_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CX 是 evidence provenance verifier，不重新训练，不调 threshold。
- 所有 artifact 均标注 fresh_run、cached_verified、not_run 或 unknown_source_label。
- worktree dirty/untracked 状态会被记录为 provenance caveat，不会被隐藏。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- artifacts_total: `25`
- artifacts_gate_passed: `25`
- source_label_counts: `{'fresh_run': 24, 'cached_verified': 1}`
- artifacts_with_worktree_caveat: `0`
- paper_files_with_worktree_caveat: `0`

## Artifact Matrix

| claim area | role | source label | gate | runner | json status | md status | caveat |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| `data_calibration` | data/metric-time boundary | `fresh_run` | `7/7` | `run_stage42_data_calibration.py` | `clean` | `clean` | `False` |
| `external_validation` | external raw-frame validation | `fresh_run` | `10/10` | `run_stage42_external_validation.py` | `clean` | `clean` | `False` |
| `full_waypoint_dynamics` | full-waypoint all-agent dynamics | `fresh_run` | `12/12` | `run_stage42_full_waypoint_dynamics.py` | `clean` | `clean` | `False` |
| `causal_ablation` | causal component ablation | `fresh_run` | `12/12` | `run_stage42_causal_ablation.py` | `clean` | `clean` | `False` |
| `safety_floor` | teacher/Stage37 floor necessity | `fresh_run` | `12/12` | `run_stage42_safety_floor.py` | `clean` | `clean` | `False` |
| `paper_package` | paper package scaffold | `fresh_run` | `12/12` | `run_stage42_paper_package.py` | `clean` | `clean` | `False` |
| `strict_time_geometry_calibration` | strict source time/geometry claim guard | `fresh_run` | `13/13` | `run_stage42_source_time_geometry_calibration.py` | `clean` | `clean` | `False` |
| `metric_time_claim_guard` | metric/seconds overclaim blocker | `fresh_run` | `11/11` | `run_stage42_metric_time_claim_guard.py` | `clean` | `clean` | `False` |
| `source_terms_validation` | legal source terms conversion gate | `fresh_run` | `11/11` | `run_stage42_source_terms_confirmation_validator.py` | `clean` | `clean` | `False` |
| `context_contribution_forensics` | context contribution boundary | `fresh_run` | `13/13` | `run_stage42_context_contribution_forensics.py` | `clean` | `clean` | `False` |
| `goal_scene_gated_expert` | goal/scene negative gated expert | `fresh_run` | `10/10` | `run_stage42_goal_scene_gated_expert.py` | `clean` | `clean` | `False` |
| `neighbor_interaction_gated_expert` | neighbor/interaction negative gated expert | `fresh_run` | `11/11` | `run_stage42_neighbor_interaction_gated_expert.py` | `clean` | `clean` | `False` |
| `common_validation_bridge_shape_composer` | endpoint/full-waypoint common-row composer | `cached_verified` | `14/14` | `run_stage42_common_validation_bridge_shape_composer.py` | `clean` | `clean` | `False` |
| `composer_safety_bootstrap` | composer bootstrap and joint safety | `fresh_run` | `14/14` | `run_stage42_common_validation_composer_safety.py` | `clean` | `clean` | `False` |
| `proximity_aware_composer_guard` | validation-only proximity guard | `fresh_run` | `19/19` | `run_stage42_proximity_aware_composer_guard.py` | `clean` | `clean` | `False` |
| `proximity_guard_ablation` | accuracy/safety Pareto ablation | `fresh_run` | `19/19` | `run_stage42_proximity_guard_ablation.py` | `clean` | `clean` | `False` |
| `frozen_proximity_guard_policy` | frozen deployable policy artifact | `fresh_run` | `25/25` | `run_stage42_freeze_proximity_guard_policy.py` | `clean` | `clean` | `False` |
| `frozen_policy_replay` | policy artifact replay verifier | `fresh_run` | `30/30` | `run_stage42_replay_proximity_guard_policy.py` | `clean` | `clean` | `False` |
| `runtime_policy_api` | runtime policy API smoke evidence | `fresh_run` | `19/19` | `run_stage42_runtime_proximity_guard_policy.py` | `clean` | `clean` | `False` |
| `batch_runtime_replay` | real batch runtime exact replay | `fresh_run` | `25/25` | `run_stage42_batch_replay_proximity_guard_policy.py` | `clean` | `clean` | `False` |
| `runtime_replay_paper_refresh` | paper/reproducibility refresh | `fresh_run` | `25/25` | `run_stage42_runtime_replay_paper_refresh.py` | `clean` | `clean` | `False` |
| `group_consistency_full_waypoint_repair` | all-agent group-consistency full-waypoint repair | `fresh_run` | `17/17` | `run_stage42_group_consistency_full_waypoint_repair.py` | `clean` | `clean` | `False` |
| `frozen_group_consistency_policy` | frozen group-consistency full-waypoint policy artifact | `fresh_run` | `22/22` | `run_stage42_freeze_group_consistency_policy.py` | `clean` | `clean` | `False` |
| `group_consistency_policy_replay` | group-consistency policy artifact replay verifier | `fresh_run` | `34/34` | `run_stage42_replay_group_consistency_policy.py` | `clean` | `clean` | `False` |
| `group_consistency_runtime_policy` | callable group-consistency full-waypoint runtime policy API | `fresh_run` | `30/30` | `run_stage42_group_consistency_runtime_policy.py` | `clean` | `clean` | `False` |

## Paper File Status

| file | exists | claim boundary | git status | caveat |
| --- | ---: | ---: | --- | ---: |
| `outputs/stage42_long_research/paper_outline_stage42.md` | `True` | `True` | `clean` | `False` |
| `outputs/stage42_long_research/method_draft_stage42.md` | `True` | `True` | `clean` | `False` |
| `outputs/stage42_long_research/experiment_tables_stage42.md` | `True` | `True` | `clean` | `False` |
| `outputs/stage42_long_research/ablation_tables_stage42.md` | `True` | `True` | `clean` | `False` |
| `outputs/stage42_long_research/failure_taxonomy_stage42.md` | `True` | `True` | `clean` | `False` |
| `outputs/stage42_long_research/model_card_stage42.md` | `True` | `True` | `clean` | `False` |
| `outputs/stage42_long_research/data_card_stage42.md` | `True` | `True` | `clean` | `False` |
| `outputs/stage42_long_research/reproducibility_stage42.md` | `True` | `True` | `clean` | `False` |
| `outputs/stage42_long_research/a_journal_gap_stage42.md` | `True` | `True` | `clean` | `False` |

## Interpretation

- Stage42-CX provides a compact provenance and command matrix for the paper package.
- Worktree caveats are intentionally visible; they are not new claims and must be resolved or cited as caveats before a frozen paper artifact release.
- The supported claim remains protected dataset-local/raw-frame 2.5D only.
