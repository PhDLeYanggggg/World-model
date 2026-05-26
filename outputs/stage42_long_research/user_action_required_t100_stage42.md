# Stage42-BB User Action Required For T100

- source: `fresh_synthesis_from_stage42_ba_and_calibration`
- purpose: list concrete data/calibration actions needed before t100 can be claimed as stable positive transfer.

## ETH_UCY

- priority: `high`
- action_type: `provide_more_independent_t100_capable_topdown_sources_or_source_specific_repair`
- reason: `t100_easy_safety_not_stable_across_source_cv`
- minimum_extra_sources: `2`
- notes: Provide legal raw/source files with train/val/test provenance, long enough t100 tracks, and official FPS/stride/homography/scale if available. Do not use test endpoints for goals or threshold selection.

## TrajNet

- priority: `high`
- action_type: `provide_more_independent_t100_capable_topdown_sources_or_source_specific_repair`
- reason: `t100_easy_safety_not_stable_across_source_cv`
- minimum_extra_sources: `1`
- notes: Provide legal raw/source files with train/val/test provenance, long enough t100 tracks, and official FPS/stride/homography/scale if available. Do not use test endpoints for goals or threshold selection.

## UCY

- priority: `high`
- action_type: `provide_more_independent_t100_capable_topdown_sources_or_source_specific_repair`
- reason: `insufficient_t100_capable_original_train_sources`
- minimum_extra_sources: `1`
- notes: Provide legal raw/source files with train/val/test provenance, long enough t100 tracks, and official FPS/stride/homography/scale if available. Do not use test endpoints for goals or threshold selection.

## sdd

- priority: `medium`
- action_type: `user_action_or_source_specific_repair_required`
- official_hint: `https://cvgl.stanford.edu/projects/uav_data/`
- reasons:
  - Can support SDD pixel raw-frame work, but does not repair external t100 source support unless separately converted/aligned.
  - keep dataset-local/raw-frame claim until official FPS/stride/homography/scale are verified.
  - do not auto-download without user accepting official terms/login/application.

## opentraj

- priority: `medium`
- action_type: `user_action_or_source_specific_repair_required`
- official_hint: `https://github.com/crowdbotp/OpenTraj`
- reasons:
  - Use OpenTraj only through legal underlying dataset terms; do not treat toolkit mirror as license override.
  - keep dataset-local/raw-frame claim until official FPS/stride/homography/scale are verified.

## eth_ucy

- priority: `high`
- action_type: `user_action_or_source_specific_repair_required`
- official_hint: `ETH/BIWI + UCY original dataset pages; verify source-specific terms`
- reasons:
  - t100 source-CV blocker: t100_easy_safety_not_stable_across_source_cv; needs at least 2 additional safe t100-capable train source(s) or source-specific repair.
  - keep dataset-local/raw-frame claim until official FPS/stride/homography/scale are verified.

## trajnet

- priority: `high`
- action_type: `user_action_or_source_specific_repair_required`
- official_hint: `https://www.epfl.ch/labs/vita/datasets/`
- reasons:
  - t100 source-CV blocker: t100_easy_safety_not_stable_across_source_cv; needs at least 1 additional safe t100-capable train source(s) or source-specific repair.
  - keep dataset-local/raw-frame claim until official FPS/stride/homography/scale are verified.

## ucy

- priority: `high`
- action_type: `user_action_or_source_specific_repair_required`
- official_hint: `http://graphics.cs.ucy.ac.cy/research/downloads/crowd-data`
- reasons:
  - t100 source-CV blocker: insufficient_t100_capable_original_train_sources; needs at least 1 additional safe t100-capable train source(s) or source-specific repair.
  - keep dataset-local/raw-frame claim until official FPS/stride/homography/scale are verified.

## tgsim

- priority: `medium`
- action_type: `user_action_or_source_specific_repair_required`
- official_hint: `https://data.transportation.gov/`
- reasons:
  - Traffic metric diagnostic only; do not use as pedestrian/drone world-model success.
  - do not auto-download without user accepting official terms/login/application.

## aerialmpt

- priority: `medium`
- action_type: `user_action_or_source_specific_repair_required`
- official_hint: `DLR AerialMPT official page required; local prior stages have derived scene packs`
- reasons:
  - Potential long aerial/top-down source, but official terms, raw sequences, FPS/stride and calibration must be verified before claims.
  - keep dataset-local/raw-frame claim until official FPS/stride/homography/scale are verified.
  - do not auto-download without user accepting official terms/login/application.

## Non-Actionable Non-Claims

- Do not use TGSIM traffic metric success as pedestrian/top-down t100 success.
- Do not use SDD pixel raw-frame success as external metric/time calibration.
- Do not use test endpoints, future waypoints, or central velocity as inference input.
- Do not write t100 as seconds-level until FPS/stride/effective seconds are verified.
