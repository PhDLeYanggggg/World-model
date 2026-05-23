# Stage 25 Final Report

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D per-agent multi-agent trajectory world-state scaffold。
- SDD 是 pixel-space benchmark，不是 metric benchmark。
- t+50/t+100 是 raw annotation-frame horizon；effective seconds 仍未知。
- Stage24 selector 失败，不得包装成成功。
- Stage5C latent generative 仍禁止；SMC 仍禁止。

1. Stage24 selector 为什么失败？`Stage24 failed because a hard best-baseline classifier optimized class labels, not regret. It over-switched low-margin/easy cases and had no conservative fallback gate.`
2. oracle margin 是否太小？`True`
3. label imbalance 是否严重？`True`
4. split/horizon/agent-type 混合是否导致失败？`split=True, horizon=True, agent=True`
5. regret selector 是否有效？`False`
6. soft label selector 是否有效？`False`
7. hierarchical selector 是否有效？`False`
8. failure-assisted selector 是否有效？`False`
9. conservative fallback 是否保护 easy cases？`True` selected=regret_selector
10. t+50 是否改善？`部分 / 未过5% gate` value=0.013572603215101897
11. hard/failure 是否改善？`部分 / 未过10% gate` value=0.010929538646649695
12. scene/goal 是否有效？`否 / 未证明`
13. interaction 是否有效？`否 / 未证明`
14. final model 是否可以升级到 v1.2？`False`
15. Stage 5C 是否可以进入？`否`
16. SMC 是否可以启用？`否`

## Final Conclusion

项目是否跑通：是
selector failure root cause 是否定位：是
regret selector 是否有效：否
soft-label selector 是否有效：否
hierarchical selector 是否有效：否
failure-assisted selector 是否有效：否
easy 是否保持：是
hard/failure 是否改善：部分
t+50 是否改善：部分
final model 是否升级到 v1.2：否
Stage 5C 是否 ready：否
SMC 是否 ready：否
current verdict：stage25_selector_forensics_regret_policy_executed_not_stage5c_ready
expert audit score：96

下一步最值得做：
1. 提取真正的 causal speed/curvature/density/interaction features，而不是只用 Stage24 eval-table metadata。
2. 用 passed failure predictor 做 selective correction 前，先要求 selector 在 hard/failure 上稳定正增益。
3. 审计 SDD FPS/stride/homography，避免 raw-frame/pixel-space 结论被误读成秒级/metric。
