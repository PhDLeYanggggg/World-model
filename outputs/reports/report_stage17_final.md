# Stage 17 Final Report

## Direct Answers

1. per-sample baseline oracle 是否有 headroom？是
2. baseline selector 是否训练成功？部分
3. selector 是否超过 global strongest baseline？official t+50 improvement = `0.081954`。
4. correction specialist 是否有额外提升？`0.000000`。
5. hard/failure 是否改善？`0.040700`。
6. easy subset 是否保持？`True`。
7. scene/goal 是否有贡献？否/未证明。
8. interaction 是否有贡献？部分。
9. final model v1.1 是否优于 BPSG-MA v1？部分。
10. 是否仍禁止 Stage 5C？是。
11. 是否仍禁止 SMC？是。
12. 下一步是继续模型，还是必须补数据/标注？优先补 SDD/OpenTraj 数据和 human-confirmed annotations。

## Final Conclusion

项目是否跑通：是
baseline selector oracle 是否有 headroom：是
baseline selector 是否有效：是
correction specialist 是否有效：否
official t+50 是否改善：部分
hard/failure 是否改善：部分
easy 是否保持：是
scene/goal 是否有效：否/未证明
interaction 是否有效：部分
是否优于 BPSG-MA v1：部分
latent generative Stage 5C 是否 ready：否
SMC 是否 ready：否
current verdict：stage17_selector_v1_1_candidate_diagnostic
expert audit score：89

下一步最值得做：
- Provide SDD/OpenTraj local paths and convert them legally.
- Human-confirm high-value annotation tasks into silver/gold labels.
- Expand official pedestrian/drone t+100 and hard/failure rows before more correction training.
