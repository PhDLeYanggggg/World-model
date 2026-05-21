# Stage 12 Horizon Audit

| dataset_name | original_fps | dt_seconds | raw_frame_horizon | t10_seconds | t25_seconds | t50_seconds | t100_seconds | samples_t50 | samples_t100 | whether_downsampling_used | whether_horizon_is_raw_or_downsampled | whether_official_verified | usable_for_stage13_training |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eth_ucy_ewap | 2.5 | 0.4 | 190 | 4.0 | 10.0 | 20.0 | 40.0 | 13 | 1 | False | raw_annotation_steps | True | True |
| aerialmpt | unknown | unknown | 30 | unknown | unknown | unknown_frame_seconds | unknown_frame_seconds | 0 | 0 | False | raw_annotation_steps | False | True |
| full_trajnet_original_quick | unknown | unknown | 20 | unknown | unknown | None | None | 0 | 0 | False | raw_annotation_steps | False | True |
| stanford_drone_dataset | unknown | unknown | 0 | unknown | unknown | None | None | 0 | 0 | False | raw_annotation_steps | False | False |
| opentraj | unknown | unknown | 0 | unknown | unknown | None | None | 0 | 0 | False | raw_annotation_steps | False | False |

Only actual verified t+50/t+100 sources count for the long-horizon gate; no t+10 data is relabeled as long horizon.
