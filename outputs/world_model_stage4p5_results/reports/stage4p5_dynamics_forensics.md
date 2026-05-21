# Stage 4.5 Dynamics Forensics

## Required Admission

TGSIM Foggy Bottom 已经接入，并已经构建 verified real t+100 episodes。Stage 4 项目跑通但只通过 2/7 gates，expert_audit_score 仍是 58/100，verdict 仍是 prototype_with_major_failures。constant_velocity_baseline 是当前最强模型；hand_physics、learned residual、SMC 都没有超过 constant velocity。当前不应该进入 latent generative Stage 5。

## Findings

1. TGSIM 中 frame_id 到真实时间 dt 的定义是什么？
   - `frame_id` 是 dense index；真实 dt 来自原始 `time` 列。当前审计 median dt=0.09999999999999432。

2. 当前 loader 使用的 dt 是多少？
   - Stage 4.5 使用 dataset `time` 推断 dt，正式 rollout 使用每个 episode 的 `dt_seconds`，不是固定 1 frame。

3. 速度单位是什么？
   - native velocity 来自 TGSIM `speed_kf_x/speed_kf_y`，按 m/s 处理；causal_fd_velocity 用 `(x_t - x_t-1) / dt`，也是 m/s。

4. 加速度单位是什么？
   - native acceleration 来自 TGSIM `acceleration_kf_x/y`，按 m/s^2 处理；causal acceleration 从 causal velocity 的 past-only 差分得到。

5. 当前模型 rollout 使用的 dt 是否和数据 dt 一致？
   - Stage 4.5 是；Stage 4 不是完全可靠，因为旧 rollout 等价把 frame step 当作 dt=1。

6. constant velocity 是否使用 dataset-native velocity？
   - Stage 4 主要使用 loader 提供的 velocity。Stage 4.5 将 native、causal_fd、central_fd 分开报告，正式 benchmark 默认 causal_fd。

7. dataset-native velocity 是否可能经过全轨迹平滑，从而包含未来信息？
   - 有风险。TGSIM 列名包含 `_kf`，可能来自 Kalman smoothing/filtering；因此 native 只做对照，不作为正式 causal score。

8. 当前 finite difference velocity 是否使用了未来帧？
   - `causal_fd_velocity` 不使用未来帧；`central_fd_velocity` 使用 t+1，只作为 diagnostic。

9. 如果使用 central difference？
   - 视为潜在 future leakage，不进入正式预测输入。

10. hand physics 的 t+1 误差为什么接近 1m，而 constant velocity 的 t+1 误差只有约 0.009m？
   - Stage 4 hand physics 在没有真实 goal/scene geometry 的 TGSIM 上仍施加 goal attraction，使近乎静止或平滑行驶的轨迹被推向人工目标，t+1 立即偏移约 1m。

11. hand physics 是否错误地施加了 social force、goal force、boundary force 或 scene clamp？
   - 是。Stage 4 的默认 social-force-like dynamics 对 TGSIM quick endpoint 不适配；Stage 4.5 默认关闭 goal/social/obstacle/boundary force。

12. TGSIM 主要包含车辆、行人，还是混合 agent type？
   - 当前 quick endpoint 主类型为 `3.0`。这更像 traffic/generic trajectory benchmark，而不是纯 human crowd benchmark。

13. 如果是车辆轨迹，当前 human crowd social-force model 是否不适配？
   - 是。应该改为 traffic kinematic world model，或仅把 TGSIM 当 generic trajectory dynamics benchmark。

14. learned residual 的 target 是否建立在错误 hand physics 上？
   - Stage 4 是。Stage 4.5 新增 residual over constant_velocity / constant_acceleration / tuned_hand_physics。

15. SMC proposal 是否只是局部噪声，没有真实 intent / route proposal？
   - 是。Stage 4.5 在 deterministic learned model 不具备竞争力前，把 SMC gate 标为 premature。

## Conclusion

当前失败主要是数据/单位/动力学设定问题，同时也有模型学习问题。最先要修的是 causal dt/velocity、无真实 scene geometry 时的 force 开关，以及 residual target；不是直接加大模型容量。
