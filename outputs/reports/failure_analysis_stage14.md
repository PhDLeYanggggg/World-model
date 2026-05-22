# Stage 14 Failure Analysis

- Stage 13 的主要问题是 per-agent causal mask 下 EWAP t+100 rows 不可评估；Stage14 已重建独立 t+100 episodes 并显式记录 mask。
- Deterministic improvement 仍必须被 strongest causal baseline 审判；若 improvement 未达阈值，不允许进入 Stage 5C。
- Visual/raster scene features 当前仍是轻量 multimodal scaffold，不能称为已证明有效。
- Runtime note: two earlier Stage14 attempts triggered the local Apple Silicon OpenMP/SHM issue before heartbeat startup. The fixed runner avoids torch resource probing and executes Stage14 core tasks inline. The old PIDs still appear unkillable after SIGKILL, so a machine reboot may be needed to clear those stale processes.
