# Stage42-Z Paper Claim Evidence Audit

- source: `fresh_audit_from_stage42_wxy_and_paper_package_artifacts`
- generated_at_utc: `2026-05-26T05:06:22.224854+00:00`
- git_commit: `b1280f8`
- input_hash: `4ad0fd249b895f0e7bd7cde85f0f661430f9ddb9674a5645e2d4c8c628e2bf03`
- gate: `16 / 16`
- verdict: `stage42_z_paper_claim_evidence_audit_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- SDD 是 pixel-space；external 是 dataset-local / unverified weak metric diagnostic。
- t+50 / t+100 是 raw-frame horizon，不能写成 seconds-level。
- Stage42-Z 是 claim-to-evidence audit，不重新训练大模型，不读取 raw data/cache。
- future endpoints / waypoints 只可作为 label 或 evaluation，不可作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test 调阈值。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Claim Matrix

| id | claim | status | source | main claim? | evidence |
| --- | --- | --- | --- | --- | --- |
| `C1` | M3W has a unified row-level external full-waypoint 2.5D evidence cache over ETH_UCY, TrajNet, and UCY. | `supported_fresh` | `fresh_run_from_stage42s_row_cache_and_stage42v_ucy_predictions` | `True` | Stage42-X gates 16/16; domains=['ETH_UCY', 'TrajNet', 'UCY']; ADE all=0.090014; t50=0.061094; hard=0.093746; easy=0.001102 |
| `C2` | External t50 full-waypoint evidence is bootstrap/seed positive under raw-frame dataset-local evaluation. | `supported_fresh` | `fresh_run_from_stage42s_row_cache_and_stage42v_ucy_predictions` | `True` | positive_t50_domains=['ETH_UCY', 'TrajNet', 'UCY']; seed_ci_low=0.053671; row_bootstrap_ci_low=0.027880 |
| `C3` | Removing the UCY full-waypoint source hurts unified t50/hard performance. | `supported_fresh_synthesis` | `fresh_synthesis_from_stage42x_row_cache_and_retrained_ablation_reports` | `True` | Stage42-Y gates 13/13; loss_if_removed_t50=0.023159; loss_if_removed_hard=0.038954 |
| `C4` | History tokens are the strongest proven retrained sequence component; domain expert helps. | `supported_fresh` | `fresh_synthesis_from_stage42x_row_cache_and_retrained_ablation_reports` | `True` | history t50 contribution=0.457817; history hard=0.470799; domain t50=0.041885 |
| `C5` | Goal/scene and neighbor/interaction contributions are established as uniformly positive. | `mixed_not_main_claim` | `fresh_synthesis_from_stage42x_row_cache_and_retrained_ablation_reports` | `False` | goal/scene t50=-0.004259; neighbor all=-0.000078; neighbor hard=-0.001343 |
| `C6` | Ungated neural can replace the Stage37/teacher safety floor. | `rejected_by_evidence` | `fresh_run` | `False` | ungated easy degradation=1.245861; floor conclusion=teacher_floor_required_for_current_deployment |
| `C7` | Protected endpoint/composite-tail external validation remains a strong deployable floor. | `supported_fresh` | `fresh_run` | `True` | all=0.210251; t50=0.136522; t100diag=0.146941; hard=0.203849; easy=-0.145111 |
| `C8` | Stage42-C full-waypoint sequence dynamics has positive evidence on at least two external domains. | `supported_fresh_but_protected` | `fresh_run` | `True` | Stage42-C gates 12/12; positive_domains=['ETH_UCY', 'TrajNet']; ADE all=0.185779; t50=0.148037 |
| `C9` | Metric or seconds-level pedestrian world-model claims are supported. | `not_supported` | `fresh_run` | `False` | global_metric_claim_allowed=False; global_seconds_claim_allowed=False |
| `C10` | M3W is a true 3D or foundation world model. | `not_supported` | `claim_boundary` | `False` | Stage42 claim boundaries keep true_3d=false and foundation_world_model=false. |
| `C11` | A-journal evidence package is complete enough to draft a protected 2.5D paper, but not enough for broad foundation/3D claims. | `supported_as_gap_aware_package` | `fresh_run` | `True` | paper package claims=7; paper final verdict=stage42_f_paper_package_complete_not_full_a_journal_ready; Stage42-Z keeps non-claims explicit. |

## Paper Files

| file | exists | size_bytes |
| --- | --- | ---: |
| `outputs/stage42_long_research/paper_outline_stage42.md` | `True` | 2794 |
| `outputs/stage42_long_research/method_draft_stage42.md` | `True` | 1871 |
| `outputs/stage42_long_research/experiment_tables_stage42.md` | `True` | 796 |
| `outputs/stage42_long_research/ablation_tables_stage42.md` | `True` | 8047 |
| `outputs/stage42_long_research/failure_taxonomy_stage42.md` | `True` | 1379 |
| `outputs/stage42_long_research/model_card_stage42.md` | `True` | 913 |
| `outputs/stage42_long_research/data_card_stage42.md` | `True` | 731 |
| `outputs/stage42_long_research/reproducibility_stage42.md` | `True` | 802 |
| `outputs/stage42_long_research/a_journal_gap_stage42.md` | `True` | 5029 |

## Interpretation

- Stage42-X/Stage42-Y are now the strongest row-level full-waypoint and ablation evidence anchors.
- The paper-ready claim is a protected, dataset-local raw-frame 2.5D world-state candidate, not true 3D/foundation/metric/seconds-level.
- UCY full-waypoint source contribution and history-token contribution are supported; goal/scene and neighbor/interaction evidence is mixed and should be written as limitation or partial evidence.
- The Stage37/teacher floor remains necessary; ungated neural is rejected for deployment safety.
- Stage5C and SMC remain disabled.
