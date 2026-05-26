# Stage42-AJ Post-Repair Paper Package Refresh

- source: `fresh_synthesis_from_stage42_ad_to_ai_artifacts`
- generated_at_utc: `2026-05-26T06:49:30.250317+00:00`
- git_commit: `e20dd36`
- gate: `10 / 10`
- verdict: `stage42_aj_post_repair_paper_package_refresh_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-AJ 刷新 paper package，纳入 Stage42-AD 到 Stage42-AI，不重新训练模型。
- future endpoints / waypoints 只可作为 label 或 evaluation，不可作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test 调阈值。
- t+50 / t+100 是 raw-frame horizons，不能说成 seconds-level。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Refreshed Evidence Rows

| item | status | paper use | evidence |
| --- | --- | --- | --- |
| Stage42-AD calibration evidence refresh | `stage42_ad_calibration_evidence_refresh_pass` | data/calibration boundary | audited=7, files=1152, metric_allowed=False, seconds_allowed=False |
| Stage42-AF horizon25 validation-margin guard | `stage42_af_weak_slice_guard_repair_pass_with_eth_t50_limitation` | weak-slice safety repair | horizon25 -0.004781149088858072 -> 0.0; validation-only low-margin guard |
| Stage42-AG ETH_UCY t50/FDE source repair | `stage42_ag_eth_t50_fde_source_repair_pass` | domain t50/FDE lower-bound repair | ADE@50 low -0.013218100958604987 -> 0.002820688160982139; FDE@50 low -0.04199023614248535 -> 0.021040393452369632 |
| Stage42-AH post-repair claim matrix | `stage42_ah_post_repair_claim_refresh_pass` | claim matrix and remaining limitations | all_low=0.085258, t50_low=0.058513, hard_low=0.089767, easy_high=0.003348 |
| Stage42-AI TrajNet t100 easy-safety repair | `stage42_ai_trajnet_t100_safety_repair_pass` | raw-frame diagnostic t100 safety repair | TrajNet100 easy high 0.08498424090178214 -> 0.0; global t100 raw-frame low=0.068349 |

## Paper Files

| file | refreshed | no-overclaim boundary |
| --- | ---: | ---: |
| `outputs/stage42_long_research/paper_outline_stage42.md` | `True` | `True` |
| `outputs/stage42_long_research/method_draft_stage42.md` | `True` | `True` |
| `outputs/stage42_long_research/experiment_tables_stage42.md` | `True` | `True` |
| `outputs/stage42_long_research/ablation_tables_stage42.md` | `True` | `True` |
| `outputs/stage42_long_research/failure_taxonomy_stage42.md` | `True` | `True` |
| `outputs/stage42_long_research/model_card_stage42.md` | `True` | `True` |
| `outputs/stage42_long_research/data_card_stage42.md` | `True` | `True` |
| `outputs/stage42_long_research/reproducibility_stage42.md` | `True` | `True` |
| `outputs/stage42_long_research/a_journal_gap_stage42.md` | `True` | `True` |

## Verdict

The Stage42 paper package now includes AD-AI calibration, stress, safety repair, and post-repair claim boundary evidence. The paper-ready scope is stronger than Stage42-AC, but remains protected dataset-local raw-frame 2.5D. Metric, seconds-level, true-3D, foundation, Stage5C, SMC, and ungated neural deployment claims remain rejected.
