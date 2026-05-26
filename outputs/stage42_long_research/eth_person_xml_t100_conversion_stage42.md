# Stage42-BL ETH-Person XML T100 Conversion Dry-Run

- source: `fresh_technical_dry_run_terms_unverified`
- generated_at_utc: `2026-05-26T13:34:42.379027+00:00`
- git_commit: `028b112`
- input_hash: `8d6849e0ddf464fb5ffe4241394c9242192c6f5fa6fcc648bc56bbce670c9f65`
- gate: `13 / 13`
- verdict: `stage42_bl_eth_person_xml_t100_dry_run_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BL 是 ETH-Person XML 技术转换 dry-run 与 train-only source-CV，不是 official dataset claim。
- ETH-Person local XML license/terms 尚未由用户确认，因此结果标记为 technical_dry_run_terms_unverified。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- 不写 materialized feature store，不提交 raw XML/data/cache。
- t100 仍是 raw-frame diagnostic，不能写成 seconds-level。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Candidate Sources

| source_id | relative_path | independent_key | rows | agents | max track | t100 windows | license |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `ETH-Person_bahnhof_assc_gt` | `ETH-Person/data/bahnhof_assc_gt.xml` | `ETH_UCY::ETH-Person/bahnhof_assc_gt` | 7653 | 223 | 217 | 377 | `local_path_present_terms_unverified` |
| `ETH-Person_jelmoli_assc_gt` | `ETH-Person/data/jelmoli_assc_gt.xml` | `ETH_UCY::ETH-Person/jelmoli_assc_gt` | 2582 | 74 | 161 | 136 | `local_path_present_terms_unverified` |
| `ETH-Person_seq0_assc_gt` | `ETH-Person/data/seq0_assc_gt-interp.xml` | `ETH_UCY::ETH-Person/seq0_assc_gt` | 2573 | 46 | 353 | 471 | `local_path_present_terms_unverified` |
| `ETH-Person_sunnyday_assc_gt` | `ETH-Person/data/sunnyday_assc_gt.xml` | `ETH_UCY::ETH-Person/sunnyday_assc_gt` | 1898 | 36 | 305 | 410 | `local_path_present_terms_unverified` |
| `ETH_seq_eth` | `ETH/seq_eth/obsmat.txt` | `ETH_UCY::ETH/seq_eth` | 8908 | 360 | 190 | 91 | `local_path_present_terms_unverified` |

## Source-CV Technical Dry-Run

| holdout | validation | h50 safe | h50 gain | h100 safe | h100 gain | h100 easy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ETH-Person_seq0_assc_gt` | `ETH-Person_sunnyday_assc_gt` | `True` | 0.4338597553820991 | `True` | 0.7211316232637657 | -0.6392294227023367 |
| `ETH-Person_sunnyday_assc_gt` | `ETH-Person_seq0_assc_gt` | `True` | 0.7781065937048592 | `True` | 0.9323876101796544 | -0.8795841441215161 |
| `ETH-Person_bahnhof_assc_gt` | `ETH-Person_seq0_assc_gt` | `True` | 0.4778488117321853 | `True` | 0.6242912289570461 | -0.37465413510527557 |
| `ETH-Person_jelmoli_assc_gt` | `ETH-Person_seq0_assc_gt` | `False` | 0.35794924147695434 | `True` | 0.4964237263017553 | -0.014155078403600982 |
| `ETH_seq_eth` | `ETH-Person_seq0_assc_gt` | `False` | 0.3372511433002337 | `True` | 0.6435130945031498 | -0.06467064475829457 |

## Summary

- candidate_sources: `5`
- strict_independent_sources: `5`
- eth_person_xml_sources: `4`
- t100_windows_total: `1485`
- source_cv_folds: `5`
- technical_t100_all_folds_safe_positive: `True`
- technical_t100_mean_improvement_vs_fallback: `0.6835494566410742`
- technical_t100_min_improvement_vs_fallback: `0.4964237263017553`
- technical_t100_max_easy_degradation: `-0.014155078403600982`
- license_terms_confirmed: `False`
- official_converted_dataset_claim_allowed: `False`
- deployable_t100_claim_allowed: `False`

## Interpretation

- This dry-run proves the local ETH-Person XML loader and strict source-CV pipeline are technically executable.
- Because license/terms are still unconfirmed, these XML sources are not counted as official converted/evaluated data.
- The result cannot be used as a global t100 deployment claim until terms are confirmed and the official conversion/no-leakage/source-CV protocol is rerun.
