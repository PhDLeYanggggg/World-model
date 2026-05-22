# Stage 15 Final Report

## Direct Answers

1. continuous loop 是否真正执行到 min-hours / min-trials：是
2. EWAP t+100 rows 是否扩展：81
3. t+100 是否可 official 评估：部分/diagnostic 或小样本
4. oracle diagnostics 是否显示有学习空间：True
5. 是否训练 Stage15 deterministic model：是
6. 是否超过 strongest causal baseline：0.008001
7. 是否改善 hard/failure：hard=7.5e-05; failure=7.5e-05
8. 是否保持 easy subset：True
9. scene/goal 是否有效：False
10. interaction 是否有效：False
11. 是否接入更多 multimodal data：partial
12. 是否需要用户提供 SDD/OpenTraj：是，若要扩大真实 multimodal pedestrian/drone 数据。
13. Stage 5C 是否 ready：否
14. SMC 是否 ready：否

## Final Conclusion

项目是否跑通：是
continuous loop 是否真实执行：是
EWAP t+100 mask 是否足够：部分
oracle headroom 是否存在：是
deterministic model 是否超过 strongest causal baseline：否/部分
hard/failure 是否改善：hard=7.5e-05; failure=7.5e-05
easy 是否保持：是
scene/goal 是否有效：否/未证明
interaction 是否有效：否/未证明
新增 multimodal data：partial
latent generative Stage 5C 是否 ready：否
SMC 是否 ready：否
current verdict：stage15_oracle_and_deterministic_repair_executed_not_stage5c_ready
expert audit score：86

需要用户提供：
- SDD 本地路径（接受 non-commercial terms 后）。
- OpenTraj/full pedestrian-drone 数据路径。
- 对关键场景的人工确认标注。

下一步自动任务：
- Expand real pedestrian/drone long-horizon data beyond EWAP single-track limitations.
- Add human-confirmed scene/goal labels for failure-rich scenes.
- Re-run conservative repair only if oracle/headroom remains above threshold on larger data.
