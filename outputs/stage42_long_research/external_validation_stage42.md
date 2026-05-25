# Stage42-B External Validation

- source: `fresh_run`
- generated_at_utc: `2026-05-25T19:41:26.109540+00:00`
- git_commit: `2910e00`
- input_hash: `a673cabd8fc55c61b59f07bfbdb1b8f406685d04616c37daad73fb7fc8474228`

## Claim Boundary

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- External validation 使用 dataset-local raw-frame，不能写成 metric 或 seconds-level。
- future endpoint / future waypoints 只作为 label/eval，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Rebuilt Source-Level Split

- proposed_source_overlap_pass: `True`
- frozen_eval_pool_rows: `66303`
- frozen_eval_source_groups: `3`
- frozen eval protocol: old training rows are excluded for frozen-model evaluation; source files are regrouped into fresh folds inside the held-out eval pool.

## Candidate Comparison

| candidate | source | rows | all | t50 | t100 diag | hard/failure | easy degradation | switch rate | deployable note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `strongest_causal_baseline_or_stage37_floor` | `fresh_run` | 55528 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0000 | 0.0000 | diagnostic/baseline |
| `teacher_repair_floor` | `fresh_run` | 55528 | 0.2036 | 0.1312 | 0.1337 | 0.1966 | -0.1445 | 0.2954 | diagnostic/baseline |
| `m3w_neural_v1_composite_tail_protected` | `fresh_run` | 55528 | 0.2103 | 0.1365 | 0.1469 | 0.2038 | -0.1451 | 0.3410 | protected current candidate |
| `ungated_neural_endpoint` | `fresh_run` | 55528 | 0.2966 | 0.2152 | 0.3592 | 0.3294 | 1.2459 | 1.0000 | not deployable: easy safety failure |
| `oracle_floor_vs_neural_diagnostic` | `fresh_run` | 55528 | 0.4222 | 0.3452 | 0.4260 | 0.4211 | -0.2999 | 0.5740 | diagnostic/baseline |

## Protected M3W-Neural v1 By Domain

| domain | rows | all | t50 | t100 diag | hard/failure | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | 25901 | 0.1906 | 0.1358 | 0.1346 | 0.1871 | -0.1361 | 0.3938 |
| `TrajNet` | 20087 | 0.2295 | 0.1662 | 0.1402 | 0.2196 | -0.1497 | 0.2765 |
| `UCY` | 9540 | 0.2327 | 0.0914 | 0.1926 | 0.2260 | -0.1538 | 0.3334 |

## Bootstrap CI For Protected M3W-Neural v1

| slice | low | mid | high | n |
| --- | ---: | ---: | ---: | ---: |
| `all` | 0.2065 | 0.2103 | 0.2139 | 55528 |
| `t50` | 0.1307 | 0.1365 | 0.1425 | 13689 |
| `t100` | 0.1398 | 0.1469 | 0.1539 | 9905 |
| `hard_failure` | 0.2001 | 0.2039 | 0.2079 | 41741 |

## Source / Scene / Agent Stress Slices

Top source/scene/agent rows are stored in the JSON report to keep this Markdown readable. They include row counts and all/t50/t100/hard/easy/switch metrics.

## Failure Diagnosis

- {'domain': 'all', 'reason': 'ungated neural remains unsafe; keep safety floor and safe-switch', 'easy_degradation': 1.2458611044726973}
- Ungated neural remains a safety failure if easy degradation exceeds 2%; keep Stage37/teacher floor.

## Verdict

`stage42_b_external_validation_pass_protected_neural_not_ungated`
