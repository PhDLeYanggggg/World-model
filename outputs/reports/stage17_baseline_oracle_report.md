# Stage 17 Per-Sample Baseline Oracle

当前不是 true 3D world model，也不是 foundation world model；本报告只是 causal baseline selection diagnostic。

- oracle rows: `564`
- test rows: `115`
- official t+50 oracle selector improvement: `0.271291`
- diagnostic t+100 oracle selector improvement: `0.106599`
- HardBench oracle selector improvement: `0.236300`
- BaselineFailureBench oracle selector improvement: `0.271685`
- selector training worth doing: `True`

Best baseline choice distribution:
- constant_position: 373
- constant_turn_rate_velocity: 88
- constant_velocity_causal_fd: 1
- damped_velocity: 59
- route_corridor_baseline: 19
- scene_clamped_baseline: 24
