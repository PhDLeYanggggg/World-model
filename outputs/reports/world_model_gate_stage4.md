# Stage 4 World Model Gates

Passed: `2/7`
Overall pass: `False`

| Gate | Pass | Evidence | Explanation | Next Fix |
| --- | --- | --- | --- | --- |
| Real Data Gate | True | `{"total_tracks": 119, "total_frames": 24941, "dataset_name": "TGSIM Foggy Bottom"}` | 至少一个真实长轨迹数据源被读取并产生 trajectory table。 | 提供 TGSIM/TrajNet/ETH/SDD 路径并确认 loader 解析出 tracks。 |
| Verified Horizon Gate | True | `{"samples_t100": 482, "whether_t100_verified": true}` | 真实数据可构建 t+100 supervised windows。 | 使用更长轨迹或降低采样间隔；不能用 free-run 代替真值。 |
| Learned Dynamics Gate | False | `{"hand_ADE100": 17.63117, "best_learned_ADE100": 25.6339, "ADE_gain": -0.45389670679824423, "hand_FDE100": 20.05663, "best_learned_FDE100": 33.00006, "FDE_gain": -0.6453442078754008}` | learned residual 在真实 t+100 上至少比 hand physics 好 5%。 | 训练真实数据 residual；加入多步 rollout loss；不要只优化 one-step。 |
| Coverage Gate | False | `{"hand_coverage_FDE_lt_5m": 0.0, "smc_coverage_FDE_lt_5m": 0.0}` | 多分支模型提高 coverage_FDE_lt_5m 或 minFDE@N。 | 让 SMC particle 表达 goal/intent，而不是只加局部噪声。 |
| Physical Validity Gate | False | `{"hand_physical_validity": 1.0, "learned_physical_validity": 0.34653}` | learned model 不应明显恶化 collision/boundary/obstacle violation。 | 把 projection cost 和真实几何约束纳入 loss/weight。 |
| Semantic Diversity Gate | False | `{"cluster_diversity_score": 0.3339, "semantic_event_accuracy": null}` | terminal clusters 至少产生 3 个可信语义模式。 | 真实数据需要语义标签或可解释事件特征；否则只能报告 diversity，不能报告 event accuracy。 |
| Audit Score Gate | False | `{"expert_audit_score": 58.0, "target": 70}` | expert audit score >= 70。 | 先过真实 t+100、learned dynamics、coverage 三个硬门槛。 |
