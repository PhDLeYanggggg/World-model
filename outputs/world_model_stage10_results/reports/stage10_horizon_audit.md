# Stage 10 Horizon Audit

| dataset_name | original_fps | dt_seconds | raw_frame_horizon | physical_time_t10 | physical_time_t25 | physical_time_t50 | physical_time_t100 | track_count | mean_track_length | t50_sample_count | t100_sample_count | downsampling_used | horizon_is_raw_or_downsampled | downsampling_loses_interaction_detail | official_verified_t50 | official_verified_t100 | usable_for_stage11_training | honest_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sdd | None | None | 0 | None | None | None | None | 0 | 0.0 | 0 | 0 | False | raw_converted_rows | not_applicable | False | False | False | No local converted world_state found. Provide a local path or run a legal download/prepare step. |
| opentraj | None | None | 0 | None | None | None | None | 0 | 0.0 | 0 | 0 | False | raw_converted_rows | not_applicable | False | False | False | No local converted world_state found. Provide a local path or run a legal download/prepare step. |
| trajnet | 0.083333 | 12.0 | 20 | 120.0 | 300.0 | 600.0 | 1200.0 | 7500 | 20.0 | 0 | 0 | False | raw_converted_rows | not_applicable | False | False | True | Only short-horizon pedestrian/drone evaluation is available; do not claim t+50/t+100. |
| eth_ucy | 0.1 | 10.0 | 20 | 100.0 | 250.0 | 500.0 | 1000.0 | 145 | 20.0 | 0 | 0 | False | raw_converted_rows | not_applicable | False | False | True | Only short-horizon pedestrian/drone evaluation is available; do not claim t+50/t+100. |
| aerialmpt_long | None | None | 0 | None | None | None | None | 0 | 0.0 | 0 | 0 | False | raw_converted_rows | not_applicable | False | False | False | No local converted world_state found. Provide a local path or run a legal download/prepare step. |
