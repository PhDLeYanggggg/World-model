# Stage42-BD User Action Required For Local T100 Conversion

- source: `fresh_local_path_inventory`

## stage42_be_convert_local_novel_t100_sources

- priority: `high`
- candidate_count: `4`
- top_candidates:
  - `ETH/seq_eth/biwi_eth_10fps.txt`
  - `UCY/students01/students001.txt`
  - `UCY/students03/obsmat_px.txt`
  - `UCY/students03/students003.txt`
- notes: Convert only after source-specific terms/provenance are verified; then rerun train-only source-CV without test metrics.

## verify_terms_before_claim

- priority: `medium`
- notes: Local availability does not imply redistribution or metric/seconds-level claim permission.
