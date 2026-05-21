# Stage 4 Failure Analysis

1. 真实数据能不能构建 t+100？
   - 能。

2. 如果不能，卡在哪里？
   - No blocker reported.

3. learned residual 是否超过 hand physics？
   - 否。

4. 如果没有，为什么？
   - best learned residual 没有达到比 hand physics 好 5% 的门槛：hand FDE@100=20.05663, best learned FDE@100=33.00006。同时 constant velocity FDE@100=1.00923，说明真实 TGSIM quick benchmark 更接近平滑惯性运动，当前 social-force/goal prior 会把轨迹推偏。

5. SMC 是否提升 coverage？
   - 否。

6. 如果没有，为什么？
   - SMC 没有把真实未来纳入 5m 覆盖范围：hand coverage_FDE_lt_5m=0.0, physics_plus_neural_residual_SMC coverage_FDE_lt_5m=0.0, minFDE@16=31.96071。当前粒子主要是噪声扰动，缺少可学习的 intent/route proposal。

7. 当前变量 schema 是否真的帮助了真实数据？
   - 还没有被真实 ablation 证明。Stage 4 已经把真实 TGSIM 轨迹接入并训练 residual，但当前 best learned residual 仍未超过 hand physics，更未超过 constant velocity。

8. time-to-collision、closing speed、bottleneck score、obstacle tangent 等新变量是否有实际贡献？
   - 尚未通过真实 ablation 证明。TGSIM 当前接入结果没有 obstacle / exit / walkable scene geometry，因此 bottleneck、obstacle tangent、exit-distance 这类变量不能完整发挥作用。

9. synthetic 和 real 的 domain gap 有多大？
   - 已经能初步量化：synthetic 上的物理脚手架可以跑通 t+100，但真实 TGSIM 上 constant velocity 明显强于 hand physics 和 learned residual，说明当前 social-force prior 与真实轨迹分布不匹配。

10. 当前世界模型还差什么才算 strong world model？
   - 真实数据上的 learned dynamics > simple baselines、SMC coverage 提升、真实 scene geometry、类型/意图标注、semantic event labels，以及跨数据集验证。
