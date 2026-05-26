# Stage42-AL Source-Level Coverage and Claim-Gap Audit

- source: `fresh_synthesis_from_stage42_ak_ai_x_source_split`
- generated_at_utc: `2026-05-26T07:10:19.569187+00:00`
- git_commit: `c7f5dc6`
- input_hash: `0cea3d80aa7839394f5eb210585940b335e0e834535837e867afc904c3640c02`
- gate: `12 / 12`
- verdict: `stage42_al_source_level_coverage_audit_pass_with_full_split_eval_gap`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-AL 是 source-level split coverage / claim-gap audit，不重新训练模型。
- 它检查 Stage42-AK locked policy 的 post-repair metrics 是否可被写成完整 source-level split evaluation。
- 如果 coverage 不足，必须写 blocker，而不是把 available row-level stress 包装成 full source-level validation。
- t+50 / t+100 仍是 raw-frame horizons；t+100 仍只能 diagnostic。
- External coordinates 仍是 dataset-local / unverified weak-metric diagnostic。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Domain Coverage

| domain | train | val | proposed test | locked-policy stress rows | ratio | status | ADE all CI low | t50 CI low | easy CI high |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `ETH_UCY` | 134695 | 16103 | 0 | 25901 | n/a | `extra_available_not_in_proposed_source_test` | 0.028385 | 0.002821 | 0.000000 |
| `TrajNet` | 75287 | 7685 | 37918 | 20087 | 0.530 | `partial_coverage` | 0.069928 | 0.077573 | 0.003158 |
| `UCY` | 56763 | 0 | 9540 | 9540 | 1.000 | `exact_row_count_match` | 0.124112 | 0.048050 | 0.000000 |

## Horizon Coverage

| horizon | proposed test rows | locked-policy stress rows | ratio | status | ADE CI low | easy CI high |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 10 | 15402 | 16726 | 1.086 | `different_eval_pool` | 0.165349 | 0.000000 |
| 25 | 13470 | 15208 | 1.129 | `different_eval_pool` | 0.000000 | 0.000000 |
| 50 | 11538 | 13689 | 1.186 | `different_eval_pool` | 0.058513 | 0.004031 |
| 100 | 7048 | 9905 | 1.405 | `different_eval_pool` | 0.068349 | 0.000000 |

## Claim Table

| claim | status | evidence |
| --- | --- | --- |
| Stage42 post-repair policy has a frozen hash and source-level split hash. | `supported` | Stage42-AK policy_hash/source_split_hash exist and AK gate passed. |
| UCY locked-policy stress row count exactly matches proposed source-level test rows. | `supported` | exact_domains=['UCY'] |
| TrajNet locked-policy stress fully covers the proposed source-level test rows. | `not_supported` | partial_domains=['TrajNet'] |
| ETH_UCY post-repair stress rows are part of the proposed source-level test split. | `not_supported` | extra_available_not_in_proposed_source_test=['ETH_UCY'] |
| Current locked-policy metrics can be described as full proposed source-level split evaluation. | `rejected` | domain statuses include partial/extra pools; horizon statuses include ['t10', 't25', 't50', 't100']. |
| Current locked-policy metrics can be described as available row-level post-repair stress with explicit coverage gap. | `supported` | Coverage matrix reports exact, partial, and extra-pool domains separately. |
| Metric or seconds-level claims are allowed. | `rejected` | Stage42-AK/AJ claim boundaries reject metric/seconds-level claims. |
| Stage5C or SMC is enabled. | `rejected` | Stage5C and SMC remain false. |

## Next Actions

- Do not count `ETH_UCY` stress rows as proposed source-level test evidence; either move to diagnostic table or rebuild a split where that source is explicitly test.
- Rebuild full-waypoint prediction cache for proposed source-level `TrajNet` test rows: current stress rows 20087 vs proposed test rows 37918.
- Recompute horizon metrics on the proposed source-level test row set before claiming full source-level split horizon performance.
- Keep all current results dataset-local raw-frame 2.5D until calibration/homography/stride evidence is verified.
