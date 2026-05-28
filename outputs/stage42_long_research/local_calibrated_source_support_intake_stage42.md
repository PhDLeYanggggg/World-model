# Stage42-JN Local Calibrated Source Support Intake

- source: `fresh_stage42_jn_local_calibrated_source_support_intake`
- generated_at_utc: `2026-05-28T20:28:21.532997+00:00`
- git_commit: `94aec40`
- input_hash: `8913553323e46ba0986011160598441213869a9c65fbeb02cbeb0174dd2ece4b`
- gate: `12 / 12`
- verdict: `stage42_jn_local_calibrated_source_support_intake_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-JN searches local OpenTraj-style data for calibrated/source-support candidates after JM left ETH/UCY blocked.
- This is parseability and support-readiness evidence, not conversion into the deployable benchmark.
- Local calibration files and metric hints do not override license/terms or claim-boundary blockers.
- No internet scraping, no automatic download, no future endpoint input, no central velocity, no Stage5C, and no SMC are used.

## Summary

- decision: `candidate_sources_found_but_user_terms_required`
- parseable_candidates: `['Town-Center', 'Wild-Track', 'PETS-2009-S2L1']`
- long_horizon_candidates: `['Town-Center', 'Wild-Track', 'PETS-2009-S2L1']`
- can_help_blocked_eth_ucy: `['Town-Center', 'Wild-Track', 'PETS-2009-S2L1']`
- auto_convert_allowed: `[]`
- next_action: Ask for/record official terms for parseable candidate sources, then run guarded conversion/no-leakage before using them as support.

## Candidate Sources

| dataset | parseable | rows | agents | t50 | t100 | calibration files | metric status | legal auto-convert | role |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `Town-Center` | `True` | 71460 | 230 | 60417 | 50132 | 1 | `calibration_file_present_but_world_projection_not_integrated` | `False` | `user_action_required_before_conversion` |
| `Wild-Track` | `True` | 9518 | 313 | 2539 | 1770 | 16 | `README documents 2.5cm ground grid and camera calibration files` | `False` | `user_action_required_before_conversion` |
| `PETS-2009-S2L1` | `True` | 4650 | 19 | 3700 | 2768 | 8 | `camera_calibration_files_present_but_ground_projection_not_integrated` | `False` | `user_action_required_before_conversion` |

## Interpretation

- Local candidates with calibration or ground-coordinate hints exist, but this stage intentionally does not convert them into the deployable benchmark.
- The immediate value is source-support coverage for ETH/UCY blocked sources after terms/license confirmation and guarded conversion.
- Any metric/time claim must remain source-specific and restricted until conversion, no-leakage, and evaluation pass.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'conversion_executed': False, 'support_intake_only': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'global_metric_or_seconds_claim': False, 'raw_frame_dataset_local_main_claim': True, 'stage5c_executed': False, 'smc_enabled': False}`
