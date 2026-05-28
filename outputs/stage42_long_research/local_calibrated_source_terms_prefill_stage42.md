# Stage42-JP Local Calibrated Source Terms Prefill

- source: `fresh_stage42_jp_local_calibrated_source_terms_prefill`
- generated_at_utc: `2026-05-28T22:38:17.028397+00:00`
- git_commit: `f35adac`
- input_report: `outputs/stage42_long_research/local_calibrated_source_guarded_conversion_preflight_stage42.json`
- input_hash: `fe815d3fedd87d72995452f9e44623d9919aef1abec62c8ded0e53a17472ed0f`
- gate: `15 / 15`
- verdict: `stage42_jp_local_calibrated_source_terms_prefill_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-JP turns JO's guarded preflight into a user-fillable source/terms prefill, not permission.
- No local calibrated candidate is converted, downloaded, trained, evaluated, or counted as benchmark evidence.
- Official/source hints must be checked by the user before acceptance fields can be filled.
- Dataset-local/raw-frame and source-specific calibration hints are not global metric/seconds claims.
- Stage5C and SMC remain disabled.

## Summary

- decision: `terms_prefill_written_no_conversion_permission`
- datasets_prefilled: `3`
- official_hint_rows: `3`
- license_found_rows: `1`
- high_confidence_official_source_rows: `['Wild-Track']`
- manual_only_rows: `['Town-Center', 'Wild-Track', 'PETS-2009-S2L1']`
- conversion_ready_now: `0`

## Prefill Rows

| dataset | source confidence | preferred official/source URL | license local? | conversion ready now | status hint |
| --- | --- | --- | ---: | ---: | --- |
| `Town-Center` | `low` | `http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html` | `False` | `False` | `manual_terms_required_high_risk` |
| `Wild-Track` | `high` | `https://www.epfl.ch/labs/cvlab/data/data-wildtrack/` | `False` | `False` | `manual_terms_or_download_page_review_required` |
| `PETS-2009-S2L1` | `medium` | `http://www.cvg.reading.ac.uk/PETS2009/a.html` | `True` | `False` | `manual_terms_review_required_before_conversion` |

## Interpretation

- This stage makes the user action more concrete; it does not grant terms acceptance.
- Wild-Track and PETS have stronger official/source hints than Town-Center, but all still require user confirmation before conversion.
- Town-Center remains high-risk/manual-only because the local README says license information is unavailable and the historical official distribution is not verified here.
- Metric/time claims remain disabled even where calibration files exist.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'download_executed': False, 'conversion_executed': False, 'evaluation_executed': False, 'prefill_only': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'global_metric_or_seconds_claim': False, 'converted_external_support_source': False, 'prefill_is_permission': False, 'stage5c_executed': False, 'smc_enabled': False}`
