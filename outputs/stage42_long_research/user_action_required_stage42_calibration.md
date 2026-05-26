# Stage42-AD Calibration User Action Required

- source: `fresh_run`
- purpose: list evidence still needed before metric/time claims.

## Stanford Drone Dataset

- dataset_id: `sdd`
- official_hint: `https://cvgl.stanford.edu/projects/uav_data/`
- needed_for: `metric/time calibration claim boundary`
- reason: keep_dataset_local_raw_frame_claim

## OpenTraj

- dataset_id: `opentraj`
- official_hint: `https://github.com/crowdbotp/OpenTraj`
- needed_for: `metric/time calibration claim boundary`
- reason: manual_source_validation_required: confirm official coordinate convention, homography direction, frame rate, annotation stride, and meters-per-pixel before metric/seconds claim

## ETH/UCY

- dataset_id: `eth_ucy`
- official_hint: `ETH/BIWI + UCY original dataset pages; verify source-specific terms`
- needed_for: `metric/time calibration claim boundary`
- reason: manual_source_validation_required: confirm official coordinate convention, homography direction, frame rate, annotation stride, and meters-per-pixel before metric/seconds claim

## TrajNet++

- dataset_id: `trajnet`
- official_hint: `https://www.epfl.ch/labs/vita/datasets/`
- needed_for: `metric/time calibration claim boundary`
- reason: manual_source_validation_required: confirm official coordinate convention, homography direction, frame rate, annotation stride, and meters-per-pixel before metric/seconds claim

## UCY Crowd

- dataset_id: `ucy`
- official_hint: `http://graphics.cs.ucy.ac.cy/research/downloads/crowd-data`
- needed_for: `metric/time calibration claim boundary`
- reason: manual_source_validation_required: confirm official coordinate convention, homography direction, frame rate, annotation stride, and meters-per-pixel before metric/seconds claim

## AerialMPT

- dataset_id: `aerialmpt`
- official_hint: `DLR AerialMPT official page required; local prior stages have derived scene packs`
- needed_for: `metric/time calibration claim boundary`
- reason: manual_source_validation_required: confirm official coordinate convention, homography direction, frame rate, annotation stride, and meters-per-pixel before metric/seconds claim
