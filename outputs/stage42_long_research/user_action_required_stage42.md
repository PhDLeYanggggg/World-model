# Stage42 User Action Required

- source: `fresh_run`
- purpose: list datasets that cannot be legally auto-downloaded or need user-provided local paths/terms.

## Stanford Drone Dataset

- official_url_or_hint: `https://cvgl.stanford.edu/projects/uav_data/`
- reason:
  - do_not_auto_download: requires terms/login/application or source-specific approval
  - keep_raw_frame_dataset_local_claim_until_fps_stride_homography_scale_verified

## TGSIM

- official_url_or_hint: `https://data.transportation.gov/`
- reason:
  - do_not_auto_download: requires terms/login/application or source-specific approval

## AerialMPT

- official_url_or_hint: `https://www.dlr.de/`
- reason:
  - do_not_auto_download: requires terms/login/application or source-specific approval
  - keep_raw_frame_dataset_local_claim_until_fps_stride_homography_scale_verified
