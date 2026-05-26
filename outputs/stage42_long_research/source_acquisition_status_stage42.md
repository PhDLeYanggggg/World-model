# Stage42-BV Source Acquisition Status / Blocker Matrix

- source: `fresh_stage42_bv_source_acquisition_status`
- generated_at_utc: `2026-05-26T15:34:18.872948+00:00`
- git_commit: `2441d75`
- input_hash: `74b044ed601b08fcfd370c34d6dbb672400b7ace6f181843d6f1fb16e73236c9`
- gate: `16 / 16`
- verdict: `stage42_bv_source_acquisition_status_pass_blockers_actionable`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BV 是 source acquisition / blocker matrix，不训练模型，不下载数据。
- OpenTraj toolkit MIT 许可证不能自动覆盖 ETH/UCY/TrajNet 等第三方数据许可。
- TrajNet++ challenge scene snippets 不能自动等同于 raw long-track t50/t100 support。
- future endpoints / waypoints 只允许作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- blockers_total: `5`
- blockers_active: `5`
- ucy_students_blocker_narrowed: `True`
- eth_seq_blocker_resolved: `False`
- trajnet_raw_long_source_resolved: `False`
- global_t100_positive_claim_allowed: `False`
- global_metric_claim_allowed: `False`
- auto_download_executed: `False`

## Blocker Matrix

| blocker | status | root cause | next action | allowed claim |
| --- | --- | --- | --- | --- |
| `ETH_seq_t50_source_support` | `blocked` | ETH-Person XML has technical same-family h50 signal, but terms remain unverified and the validation-only dry-run does not safely repair the actual ETH_seq_eth holdout. | Verify ETH-Person terms or provide an official/legal source-compatible ETH_seq long-track source; then rerun conversion, no-leakage, source-CV, and t50 policy training. | `technical_blocker_only` |
| `UCY_students_t50_source_support` | `blocked_narrowed` | students001 and students003 are t50-capable independent sources, but students002 is too short and duplicate formats cannot be counted as independent support. | Provide one more legal independent t50-capable UCY_students-family long-track source before train/val/holdout source-CV can be attempted. | `blocker_narrowed_not_positive_transfer` |
| `TrajNet_raw_long_t100_source_support` | `blocked` | Local TrajNet files parse as fixed short challenge snippets rather than raw long tracks, so they cannot repair raw-frame t100 source-CV. | Use TrajNet++ challenge data only for scene-format diagnostics, or provide legal raw long-track TrajNet-compatible sources if t100 source-CV is required. | `short_scene_diagnostic_only` |
| `ETH_UCY_global_t100_source_support` | `blocked` | ETH-Person XML t100 dry-run is technically positive but still terms-unverified; official/deployable global t100 cannot advance without permission and rerun. | Confirm official ETH-Person terms or provide permitted ETH_UCY long raw sources, then rerun official conversion and strict source-CV. | `technical_dry_run_only` |
| `global_metric_seconds_claim` | `blocked` | Only source-specific calibration evidence exists; SDD remains pixel raw-frame and external data remain dataset-local or source-specific calibrated subsets. | Report metric/time only inside explicitly source-specific calibrated subsets after conversion/eval; keep global M3W raw-frame/dataset-local. | `source_specific_subset_only` |

## Official / Source References

| id | source confidence | url | M3W use |
| --- | --- | --- | --- |
| `trajnet_epfl_official` | `official` | https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/ | official challenge/source reference only; does not by itself provide long raw t50/t100 support |
| `trajnet_aicrowd_official` | `official_challenge_page` | https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge | official scene-format reference and access/action page; local challenge snippets remain too short for raw-frame t100 repair |
| `opentraj_github` | `official_github` | https://github.com/crowdbotp/OpenTraj | toolkit and local dataset index reference; underlying dataset terms must be audited separately |
| `eth_cvl_dataset_page` | `official_eth_page` | https://vision.ee.ethz.ch/datsets.html | official ETH data/terms reference; research-purpose boundary must be preserved |
| `ucy_crowd_data_page` | `official_url_known_but_fetch_failed_in_current_web_tool` | https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data | official UCY crowd-data action target; user/manual verification required if direct access fails |

## Next Best Actions

- provide one more independent t50-capable UCY_students-family source
- verify ETH-Person terms before using XML dry-run as official evidence
- provide legal raw long TrajNet-compatible tracks if t100 source-CV is required

## Interpretation

- Stage42-BV does not repair any blocker by itself; it converts the remaining source-support problems into an executable acquisition/status matrix.
- The strongest current deployable claim remains protected dataset-local/raw-frame 2.5D evidence, not true 3D, not foundation, not metric/seconds-level.
- UCY_students is closer after BU because `students001` is now counted as a real t50-capable independent source, but one more independent students-family source is still required.
- ETH_seq and ETH_UCY t100 remain blocked by source/terms support; TrajNet long-horizon repair remains blocked by local snippet length.
