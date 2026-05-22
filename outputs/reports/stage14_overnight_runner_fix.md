# Stage 14 Overnight Runner Fix

- Added `continuous-stage14` mode.
- min_hours: `0.25`
- max_hours: `2.0`
- max_iterations: `20`
- Queue exhaustion before min-hours now triggers safe maintenance tasks or heartbeat sleep, not early termination.
- Dynamic queue can add data dry-runs, mask audits, benchmarks, gates, failure mining, and py_compile refreshes.
- Latent generative and SMC remain blocked.
