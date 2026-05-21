# Stage 4.5 Velocity Audit

| dataset_name | default_velocity_source | official_benchmark_velocity_source | central_fd_usage | dt_min | dt_median | dt_max | dt_unique_rounded | native_vs_causal_velocity_MAE | native_vs_causal_velocity_corr | native_vs_central_velocity_MAE | native_vs_central_velocity_corr | causal_speed_mean | causal_speed_p95 | causal_speed_max | causal_accel_mean | causal_accel_p95 | causal_accel_max | missing_frame_gaps | abnormal_jumps_gt_10m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TGSIM Foggy Bottom | causal_fd | causal_fd | diagnostic_only | 0.1 | 0.1 | 1350.0 | [0.1, 0.2] | 1e-05 | 1.0 | 0.0125 | 0.99988 | 0.72342 | 4.6996 | 15.96784 | 0.25425 | 1.29867 | 6.16462 | 10 | 1 |

Native velocity and central finite difference are diagnostic; official benchmark uses causal finite difference.
