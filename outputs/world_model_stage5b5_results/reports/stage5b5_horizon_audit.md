# Stage 5B.5 Horizon Audit

Downsampling is treated as a different effective physical horizon. It is not reported as raw t+100 unless explicitly supported by contiguous source tracks.

| dataset | dt_s | max_track | max_raw_horizon | raw_t50_tracks | raw_t100_tracks | supports_raw_t50 | supports_raw_t100 | t+100 seconds | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| eth_ucy | 10.0 | 20 | 10 | 0 | 0 | False | False | 1000.0 | no raw t+100; do not fake by stitching |
| tgsim | 0.1 | 15129 | 15119 | 71 | 48 | True | True | 10.0 | raw long horizon available |
| tgsim_i90 | 0.1 | 580 | 570 | 219 | 131 | True | True | 10.0 | raw long horizon available |
| trajnet | 12.0 | 20 | 10 | 0 | 0 | False | False | 1200.0 | no raw t+100; do not fake by stitching |
