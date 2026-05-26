# Stage42-BR User Action Required: Calibrated T50 Source Support

## Why This Exists

- Stage42-BQ safely guards calibrated-subset t50 to non-harm, but positive t50 folds are zero.
- Additional same-family calibrated support is needed before enabling t50 switching on unsupported source families.

## Actions

- `ETH_seq`: `confirm_eth_person_terms_then_convert_xml_candidates`; blocked_by: `ETH-Person official terms/license not confirmed`.
  - local candidate: `ETH-Person/data/bahnhof_assc_gt.xml`
  - local candidate: `ETH-Person/data/jelmoli_assc_gt.xml`
  - local candidate: `ETH-Person/data/seq0_assc_gt.xml`
  - local candidate: `ETH-Person/data/seq0_assc_gt-interp.xml`
  - local candidate: `ETH-Person/data/sunnyday_assc_gt.xml`
- `UCY_students`: `provide_or_locate_additional_source_specific_calibrated_tracks`; blocked_by: `no verified local same-family calibrated source support`.
- `UCY_zara`: `train_family_specific_t50_policy_or_add_more_validation_sources`; blocked_by: `source support exists but validation-safe t50 policy falls back to floor`.

## Non-Claims

- Do not call BQ a positive t50 transfer result.
- Do not call source-specific calibrated subsets a global metric/seconds-level M3W benchmark.
- Do not treat ETH-Person XML as official/deployable until terms are confirmed.
