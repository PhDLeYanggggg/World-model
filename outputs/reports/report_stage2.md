# Stage 2: Learned 2.5D Crowd Physics World Model

## 1. What Changed

上一阶段是 pseudo-3D physics-informed SMC scaffold。Stage 2 增加了 SyntheticPhysicalCrowd2.5D 长轨迹环境、episode-level train/val/test split、learned neural residual transition、stochastic latent residual variant、SMC proposal 对比、t+100 真值评估、物理违规指标和 semantic terminal clustering。

这个版本仍然不是 true 3D；它是 Z=0 ground-plane 上的 2.5D / pseudo-3D 人群轨迹世界模型。

## 2. SyntheticPhysicalCrowd2.5D

| episodes | frames_min | frames_max | agents_min | agents_max | storage | train_episodes | train_agents_mean | val_episodes | val_agents_mean | test_episodes | test_agents_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 160 | 160 | 10 | 18 | data/synthetic | 20 | 13.75 | 5 | 11.2 | 5 | 14.8 |

为什么需要 synthetic data：AerialMPT bauma3 当前片段只有 16 帧，从 start frame 4 出发最多真实评估到 t+12。Synthetic 数据有 160/200 帧，因此 t+100 有真实状态，不需要把 free-run 当作准确预测。

## 3. Learned Transition

训练目标是 world-coordinate residual acceleration：

```text
A_residual = A_true_next - A_hand_physics
A_final = A_hand_physics + A_neural_residual
S_next = project_constraints(integrate(S_t, A_final))
```

训练日志包含 position、velocity、acceleration、heading、goal direction、collision surrogate、obstacle/boundary surrogate、speed、acceleration、smoothness、stochastic diversity 和 KL 项。几何违规的最终可信指标以 rollout evaluation 为准。

## 4. Strict Synthetic t+100 Evaluation

Evaluation meta:

| full_test_episodes | evaluated_test_episodes | evaluated_episode_ids | evaluated_event_labels | horizon | particles_requested | particles_used_in_this_run | quick_mode | eval_selection | future_leakage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 4 | 26, 25, 27, 29 | corridor_jam, obstacle_detour, obstacle_detour, obstacle_detour | 100 | 64 | 16 | True | event-balanced quick subset, not smallest-agent-only selection | No future states are used by proposals or weights; ground truth is used only for metrics and cluster scoring. |

Metrics:

| model | ADE@1 | ADE@10 | ADE@25 | ADE@50 | ADE@100 | FDE@1 | FDE@10 | FDE@25 | FDE@50 | FDE@100 | best_of_64_ADE@100 | best_of_64_FDE@100 | branch_count | collision_violation_rate | obstacle_violation_rate | boundary_violation_rate | max_speed_violation_rate | acceleration_violation_rate | trajectory_smoothness | coverage@64 | NLL_endpoint_t100 | cluster_diversity_score | semantic_event_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant_velocity_baseline | 0.013 | 0.1522 | 0.5404 | 1.6643 | 4.2531 | 0.013 | 0.3165 | 1.3475 | 4.2882 | 9.0362 | 4.2531 | 9.0362 | 1.0 | 0.5099 | 0.0 | 0.16831 | 0.0 | 0.0 | 0.0 | 0.0 | 51.94817 | 0.0 | 0.0 |
| hand_physics_baseline | 0.0075 | 0.0282 | 0.212 | 1.0429 | 3.0493 | 0.0075 | 0.0474 | 0.691 | 3.1237 | 6.8297 | 3.0493 | 6.8297 | 1.0 | 0.00248 | 0.0 | 0.0594 | 0.0 | 0.0 | 0.14905 | 0.25 | 35.99955 | 0.0 | 0.0 |
| deterministic_neural_residual | 0.0075 | 0.0306 | 0.214 | 1.006 | 3.0483 | 0.0075 | 0.0529 | 0.6884 | 3.0048 | 7.0215 | 3.0483 | 7.0215 | 1.0 | 0.00495 | 0.0 | 0.06931 | 0.0 | 0.0 | 0.1524 | 0.25 | 36.7073 | 0.0 | 0.0 |
| stochastic_neural_residual | 0.0075 | 0.0307 | 0.2107 | 0.9948 | 3.0364 | 0.0075 | 0.0528 | 0.6785 | 2.9684 | 7.0366 | 3.0247 | 7.0051 | 8.0 | 0.00155 | 0.0 | 0.06931 | 0.0 | 0.0 | 0.1473 | 0.25 | 36.47137 | 0.0 | 0.0 |
| hand_physics_SMC | 0.0572 | 0.4491 | 1.1895 | 2.4756 | 4.5345 | 0.0572 | 0.8729 | 2.3625 | 5.0145 | 7.6402 | 4.8051 | 7.5305 | 16.0 | 0.0 | 0.0 | 0.00495 | 0.0 | 0.0 | 0.73818 | 0.0 | 11.8511 | 0.47073 | 0.5 |
| learned_neural_SMC | 0.0432 | 0.3285 | 1.2415 | 3.3777 | 7.4974 | 0.0432 | 0.7072 | 3.0442 | 7.8333 | 14.3396 | 7.4492 | 13.9543 | 16.0 | 0.07162 | 0.0 | 0.29425 | 0.0 | 0.0 | 0.2987 | 0.0 | 74.8395 | 0.0 | 0.0 |
| physics_plus_neural_residual_SMC | 0.0493 | 0.3026 | 0.8072 | 1.9617 | 4.2157 | 0.0493 | 0.5821 | 1.757 | 4.4314 | 7.766 | 4.282 | 7.8553 | 16.0 | 0.0 | 0.0 | 0.00835 | 0.0 | 0.0 | 0.70735 | 0.0 | 20.84585 | 0.74448 | 0.0 |

注意：quick demo 使用的 branch count 可能小于 64；表里的 best-of-64 字段在 quick mode 中表示“当前实际分支数下的 best-of-N”，并由 branch_count 显式标出。

## 5. Semantic Terminal Clustering

### constant_velocity_baseline

| cluster_id | semantic_label | probability_mass | representative_trajectory_id | mean_ADE@100 | mean_FDE@100 | mean_collision_rate | mean_obstacle_violation_rate | mean_boundary_violation_rate | mean_goal_reached_rate | mean_jam_duration | confidence | is_credible | explanation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | collision_risk | 0.5 | 0 | 2.6888 | 4.5483 | 0.60396 | 0.0 | 0.0 | 0.0 | 0.07143 | 0.39604 | False | Minimum gaps are small or collision projection was frequently needed. |
| 1 | physically_invalid | 0.5 | 0 | 5.81745 | 13.5241 | 0.41584 | 0.0 | 0.33663 | 0.09445 | 0.06667 | 0.24752 | False | The rollout enters obstacle or boundary-violating regions. |

### hand_physics_baseline

| cluster_id | semantic_label | probability_mass | representative_trajectory_id | mean_ADE@100 | mean_FDE@100 | mean_collision_rate | mean_obstacle_violation_rate | mean_boundary_violation_rate | mean_goal_reached_rate | mean_jam_duration | confidence | is_credible | explanation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | stalled | 0.75 | 0 | 3.10443 | 5.80287 | 0.0 | 0.0 | 0.0 | 0.12222 | 0.58649 | 1.0 | True | Agents remain below walking speed for a large fraction of the rollout. |
| 0 | physically_invalid | 0.25 | 0 | 2.8839 | 9.9101 | 0.0099 | 0.0 | 0.23762 | 0.61111 | 0.07591 | 0.75248 | False | The rollout enters obstacle or boundary-violating regions. |

### deterministic_neural_residual

| cluster_id | semantic_label | probability_mass | representative_trajectory_id | mean_ADE@100 | mean_FDE@100 | mean_collision_rate | mean_obstacle_violation_rate | mean_boundary_violation_rate | mean_goal_reached_rate | mean_jam_duration | confidence | is_credible | explanation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | stalled | 0.75 | 0 | 3.11357 | 5.94517 | 0.0 | 0.0 | 0.0 | 0.12222 | 0.58243 | 1.0 | True | Agents remain below walking speed for a large fraction of the rollout. |
| 0 | physically_invalid | 0.25 | 0 | 2.8524 | 10.2504 | 0.0198 | 0.0 | 0.27723 | 0.61111 | 0.06326 | 0.70297 | False | The rollout enters obstacle or boundary-violating regions. |

### stochastic_neural_residual

| cluster_id | semantic_label | probability_mass | representative_trajectory_id | mean_ADE@100 | mean_FDE@100 | mean_collision_rate | mean_obstacle_violation_rate | mean_boundary_violation_rate | mean_goal_reached_rate | mean_jam_duration | confidence | is_credible | explanation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | stalled | 0.75 | 0 | 3.10837 | 5.97353 | 0.0 | 0.0 | 0.0 | 0.12222 | 0.58049 | 1.0 | True | Agents remain below walking speed for a large fraction of the rollout. |
| 0 | physically_invalid | 0.25 | 0 | 2.8248 | 10.2257 | 0.00619 | 0.0 | 0.27723 | 0.61111 | 0.07199 | 0.71658 | False | The rollout enters obstacle or boundary-violating regions. |

### hand_physics_SMC

| cluster_id | semantic_label | probability_mass | representative_trajectory_id | mean_ADE@100 | mean_FDE@100 | mean_collision_rate | mean_obstacle_violation_rate | mean_boundary_violation_rate | mean_goal_reached_rate | mean_jam_duration | confidence | is_credible | explanation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | obstacle_detour | 0.47645 | 0 | 5.95077 | 10.41387 | 0.0 | 0.0 | 0.00167 | 0.52849 | 0.31044 | 0.63426 | True | Path length is substantially longer than the direct displacement. |
| 4 | stalled | 0.25 | 9 | 3.8217 | 6.2577 | 0.0 | 0.0 | 0.0 | 0.29464 | 0.52294 | 1.0 | True | Agents remain below walking speed for a large fraction of the rollout. |
| 3 | smooth_passage | 0.11965 | 2 | 6.38697 | 11.1259 | 0.0 | 0.0 | 0.0011 | 0.59012 | 0.28597 | 0.15925 | True | Most agents keep forward progress and a meaningful fraction reaches goals. |
| 2 | physically_invalid | 0.08547 | 4 | 5.6688 | 11.2789 | 0.0 | 0.0 | 0.04752 | 0.68889 | 0.20363 | 0.32564 | True | The rollout enters obstacle or boundary-violating regions. |
| 0 | collision_risk | 0.05654 | 14 | 5.7235 | 9.6992 | 0.0 | 0.0 | 0.0022 | 0.3963 | 0.34857 | 0.07505 | False | Minimum gaps are small or collision projection was frequently needed. |
| 5 | uncertain_multimodal | 0.01188 | 6 | 4.5638 | 6.1622 | 0.0 | 0.0 | 0.0 | 0.4 | 0.39604 | 0.04751 | False | No dominant semantic event passed the rule thresholds. |

### learned_neural_SMC

| cluster_id | semantic_label | probability_mass | representative_trajectory_id | mean_ADE@100 | mean_FDE@100 | mean_collision_rate | mean_obstacle_violation_rate | mean_boundary_violation_rate | mean_goal_reached_rate | mean_jam_duration | confidence | is_credible | explanation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | physically_invalid | 1.0 | 7 | 7.4919 | 14.2919 | 0.07162 | 0.0 | 0.29425 | 0.00781 | 0.12031 | 0.63413 | False | The rollout enters obstacle or boundary-violating regions. |

### physics_plus_neural_residual_SMC

| cluster_id | semantic_label | probability_mass | representative_trajectory_id | mean_ADE@100 | mean_FDE@100 | mean_collision_rate | mean_obstacle_violation_rate | mean_boundary_violation_rate | mean_goal_reached_rate | mean_jam_duration | confidence | is_credible | explanation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | stalled | 0.26372 | 8 | 4.3036 | 6.70805 | 0.0 | 0.0 | 0.0 | 0.32381 | 0.51537 | 0.52745 | True | Agents remain below walking speed for a large fraction of the rollout. |
| 3 | smooth_passage | 0.24387 | 1 | 6.1465 | 11.64987 | 0.0 | 0.0 | 0.0 | 0.52593 | 0.28012 | 0.32516 | True | Most agents keep forward progress and a meaningful fraction reaches goals. |
| 5 | uncertain_multimodal | 0.16438 | 7 | 5.14195 | 8.66325 | 0.0 | 0.0 | 0.0033 | 0.37833 | 0.39069 | 0.32769 | True | No dominant semantic event passed the rule thresholds. |
| 1 | obstacle_detour | 0.13779 | 2 | 5.42317 | 9.63983 | 0.0 | 0.0 | 0.0 | 0.36567 | 0.32686 | 0.13779 | True | Path length is substantially longer than the direct displacement. |
| 2 | physically_invalid | 0.1228 | 11 | 5.9737 | 13.4791 | 0.0 | 0.0 | 0.0473 | 0.62963 | 0.1149 | 0.46794 | True | The rollout enters obstacle or boundary-violating regions. |
| 0 | collision_risk | 0.06744 | 14 | 5.9752 | 10.0315 | 0.0 | 0.0 | 0.01155 | 0.38889 | 0.42431 | 0.26664 | True | Minimum gaps are small or collision projection was frequently needed. |

如果多个 cluster 仍落在相同语义，结论按弱多样性处理；不会把普通多分支采样包装成语义丰富预测。

## 6. AerialMPT Re-Application Limits

Synthetic data: verified t+100.

AerialMPT bauma3: verified only up to t+12: `[1, 5, 10, 12]`.

AerialMPT t+20/t+50/t+100: no ground truth in the selected sequence, qualitative free-run only. 不能报告 ADE@100/FDE@100。

## 7. Failure Cases

- 最大失败风险：learned residual 在 quick synthetic split 上可能没有超过 hand-coded physics，尤其是 long-horizon FDE。
- terminal clusters 仍可能被 collision_risk / jam 类标签吸收，说明 branch semantics 还不够强。
- synthetic dynamics 和 AerialMPT 的真实摄像机、身份跟踪、场景标注之间仍有明显 domain gap。

## Final Conclusions

项目是否跑通：是

当前模型类型：
pseudo-3D physics-informed learned residual state-space world model

是否是真 3D：
否，是 2.5D / pseudo-3D

是否是游戏预测：
否，是真实人物物理轨迹世界模型

是否已经从 scaffold 变成 learned world model：
部分是

Synthetic t+100 是否可验证：
是

Synthetic t+100 预测质量：
弱

AerialMPT t+12 预测质量：
弱

AerialMPT t+100 预测质量：
只能 qualitative free-run / 不可评估

learned neural residual 是否超过 hand physics：
部分超过

SMC 是否提升多分支 coverage：
否

物理约束是否有效：
强

terminal clusters 是否有语义差异：
强

当前最大局限：
1. AerialMPT 当前片段没有 t+100 真值，真实数据长预测不可验证。
2. homography / camera calibration 仍是弱标定，不是真 3D。
3. neural residual 训练数据来自合成物理，真实 domain transfer 还没有被证明。

下一步最值得做：
1. 接入 Stanford Drone / TrajNet++ / ETH-UCY 等更长真实轨迹。
2. 为真实场景补 ground-plane homography、walkable/obstacle polygon 和 exit/goal 标注。
3. 用真实长轨迹做 supervised residual fine-tuning，并报告 t+100 verified metrics。
