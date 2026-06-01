# Stage43-AS Data Calibration Refresh

- source: `fresh_stage43_as_data_calibration_refresh`
- result_source: `fresh_local_path_audit_plus_cached_verified_stage42_time_geometry`
- verdict: `stage43_as_data_calibration_refresh_pass`
- gate: `10 / 10`
- data calibration ready: `True`

## Current Claim Boundary

- M3W remains protected dataset-local/raw-frame 2.5D multi-agent world-state evidence.
- SDD remains pixel-space with effective seconds unknown unless source-specific audits prove otherwise.
- ETH/UCY source-specific calibration candidates exist, but global metric/seconds claims remain blocked.
- TGSIM is traffic diagnostic only and cannot be counted as pedestrian top-down world-model success.
- AerialMPT is audited as local candidate/diagnostic until source terms and geometry are verified.
- No Stage5C execution and no SMC.

## Summary

- datasets audited: `7`
- raw paths found: `7`
- converted paths found: `7`
- external domains ready from existing state: `opentraj, eth_ucy, trajnet, ucy`
- source-specific calibration candidates: `ETH_seq_eth, ETH_seq_hotel, UCY_zara01, UCY_zara02, UCY_zara03, UCY_students03`
- global metric claim allowed: `False`
- global seconds claim allowed: `False`

## Dataset Table

| dataset | raw | converted | coordinate | calibration | metric | seconds | role |
| --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| `sdd` | `True` | `True` | pixel | pixel_raw_frame_only | `False` | `False` | official_eval / supervised_training |
| `opentraj` | `True` | `True` | dataset-local mixed | calibration_files_found_but_not_validated | `False` | `False` | external top-down source hub / loader input |
| `eth_ucy` | `True` | `True` | dataset-local | calibration_files_found_but_not_validated | `False` | `False` | external_eval / supervised_training |
| `trajnet` | `True` | `True` | dataset-local | not_verified | `False` | `False` | external_eval / supervised_training |
| `ucy` | `True` | `True` | dataset-local | calibration_files_found_but_not_validated | `False` | `False` | external_eval / supervised_training |
| `tgsim` | `True` | `True` | traffic metric if source units verified by prior stage | traffic_metric_diagnostic_only | `True` | `False` | diagnostic_only |
| `aerialmpt` | `True` | `True` | unknown / derived local | not_verified | `False` | `False` | external_eval candidate / diagnostic |

## Source-Specific Calibration

- supported source count: `6`
- supported by domain: `{'ETH_UCY': 2, 'UCY': 4}`
- Interpretation: ETH/UCY source-specific timing/coordinate evidence can support restricted future audits, but it does not upgrade the global M3W claim.

## SDD Status

- coordinate unit: `pixel`
- metric claim allowed: `False`
- seconds claim allowed: `False`
- conclusion: pixel raw-frame only

## Gate

| gate | passed |
| --- | --- |
| all_required_sources_audited | `True` |
| fresh_local_path_audit_ran | `True` |
| sdd_pixel_raw_frame_guard | `True` |
| external_domains_available | `True` |
| source_specific_calibration_recorded | `True` |
| global_metric_seconds_blocked | `True` |
| tgsim_diagnostic_only | `True` |
| aerialmpt_audited_no_metric_claim | `True` |
| no_training_or_download | `True` |
| stage5c_and_smc_false | `True` |

## Decision

Stage43 can continue using existing SDD/external data under raw-frame/dataset-local language. Metric/seconds-level claims remain blocked globally; restricted ETH/UCY calibrated-subset work must be explicitly source-specific and separately gated.
