# Stage42-FQ User Action Required: H100 Source Support

- source: `fresh_stage42_h100_source_support_repair_queue`

## `TrajNet|100`

- priority: `high`
- action_type: `provide_or_confirm_official_long_trajnet_source`
- reason: local TrajNet files are short snippets and cannot provide raw-frame h100 source support
- official_source_hint: TrajNet++ / official raw long trajectory source if available; current local snippets are too short for raw-frame h100.
- do_not_count_as_completed_until: license/terms confirmed, conversion finished, no-leakage pass, train-only source-CV positive/easy-safe

## `UCY|100`

- priority: `high`
- action_type: `confirm_terms_and_convert_local_ucy_h100_support`
- reason: local candidate support exists but terms/license and conversion/no-leakage/source-CV are not confirmed
- top_candidates:
  - `UCY/zara02/obsmat.txt`
  - `UCY/zara01/obsmat.txt`
  - `UCY/students01/students001.txt`
  - `UCY/students03/obsmat.txt`
  - `UCY/students03/obsmat_px.txt`
  - `UCY/students03/students003.txt`
- do_not_count_as_completed_until: license/terms confirmed, conversion finished, no-leakage pass, train-only source-CV positive/easy-safe
