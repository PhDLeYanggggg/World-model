# Stage33 Relative Baseline Table

- source: `fresh_run`
- Target: `relative_FDE = FDE / coordinate-invariant scale`.

| domain | split | rows | strongest |
| --- | --- | ---: | --- |
| sdd | train | 40000 | damped_velocity |
| sdd | val | 20000 | scene_clamped_baseline |
| sdd | test | 100000 | damped_velocity |
| external | train | 119109 | constant_velocity_causal_fd |
| external | val | 7685 | constant_velocity_causal_fd |
| external | test | 3636 | constant_velocity_causal_fd |
