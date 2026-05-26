# Stage42-BR Calibrated T50 Source-Support Gap Audit

- source: `fresh_calibrated_t50_source_support_gap_audit`
- generated_at_utc: `2026-05-26T14:41:16.001977+00:00`
- git_commit: `8dc7263`
- input_hash: `9b684b11bb91ffbb3dfb495f0836e3ff9dd0582fb731e4449e0f509b506fc716`
- gate: `12 / 12`
- verdict: `stage42_br_calibrated_t50_source_support_gap_audit_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BR 是 t50 source-family support gap audit，不是训练或部署成功声明。
- Stage42-BQ 已把 calibrated-subset t50 负迁移守到 0，但没有产生 t50 正迁移。
- source-specific calibration evidence 不能升级为全局 metric/seconds-level M3W claim。
- ETH-Person XML 本地技术信号仍受 terms/license blocker 限制。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- families_audited: `3`
- calibrated_sources_audited: `6`
- unsupported_family_holdout_count: `3`
- families_with_additional_sources_needed: `['ETH_seq', 'UCY_students']`
- families_with_support_but_no_positive_t50: `['UCY_zara']`
- BQ all macro: `0.04238031883856982`
- BQ t50 macro/min: `0.0` / `0.0`
- BQ easy max: `-0.0`
- BQ positive_t50_fold_count: `0`

## Source-Family Support

| family | sources | additional needed | unsupported holdouts | positive t50 holdouts | blocker |
| --- | ---: | ---: | ---: | ---: | --- |
| `ETH_seq` | 2 | 1 | 2 | 0 | `insufficient_same_family_source_support` |
| `UCY_students` | 1 | 2 | 1 | 0 | `insufficient_same_family_source_support` |
| `UCY_zara` | 3 | 0 | 0 | 0 | `enough_family_sources_but_no_safe_positive_t50_policy` |

## Local Candidate Check

- ETH-Person XML candidates: `['ETH-Person/data/bahnhof_assc_gt.xml', 'ETH-Person/data/jelmoli_assc_gt.xml', 'ETH-Person/data/seq0_assc_gt.xml', 'ETH-Person/data/seq0_assc_gt-interp.xml', 'ETH-Person/data/sunnyday_assc_gt.xml']`
- ETH-Person terms verified: `False`
- ETH-Person official conversion allowed: `False`
- TrajNet local t100-capable files: `0`

## Action Items

- `ETH_seq`: confirm_eth_person_terms_then_convert_xml_candidates (priority `high`, additional_sources_needed `1`, blocked_by `ETH-Person official terms/license not confirmed`)
- `UCY_students`: provide_or_locate_additional_source_specific_calibrated_tracks (priority `high`, additional_sources_needed `2`, blocked_by `no verified local same-family calibrated source support`)
- `UCY_zara`: train_family_specific_t50_policy_or_add_more_validation_sources (priority `medium`, additional_sources_needed `0`, blocked_by `source support exists but validation-safe t50 policy falls back to floor`)

## Interpretation

- BQ did the right safety move: unsupported t50 source-families fall back instead of producing negative transfer.
- The remaining t50 blocker is evidence support, not a permission to loosen safety gates using test data.
- ETH-style support may be repairable after ETH-Person terms are confirmed; UCY_students needs additional same-family calibrated sources; UCY_zara has enough source support but no validation-safe t50-positive policy yet.
- This remains source-specific calibrated-subset evidence only; global metric/seconds-level M3W claims remain blocked.

## Claim Boundary

- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'global_metric_claim_allowed': False, 'global_seconds_claim_allowed': False, 'm3w_official_metric_seconds_claim_allowed': False, 'positive_t50_claim_allowed': False, 't50_nonharm_claim_allowed': True, 'stage5c_executed': False, 'smc_enabled': False}`
