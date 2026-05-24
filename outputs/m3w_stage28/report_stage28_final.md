# Stage28 Final Report: M3W-LAS Evidence Sprint

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。
- SDD 是 pixel-space benchmark，不是 metric benchmark。
- t+50/t+100 是 raw annotation-frame horizon；effective seconds、homography、metric scale 未验证。
- self-audited / visual-prior labels 不是 human gold。
- Stage5C latent generative 仍禁止；SMC 仍禁止。

## Main Result

- Stage26 reference t+50 / hard / easy: `0.14583655843823773` / `0.11234058960663984` / `0.01808836280803794`
- Stage28 best variant: `all_latent`
- Stage28 t+50 improvement: `0.1686288243790961`
- Stage28 hard/failure improvement: `0.1336398986813968`
- Stage28 easy degradation: `0.01928694490688554`
- final model v2 candidate: `True`
- ablation nuance: goal and interaction features show measurable contribution; scene-only contribution is not stable on this run.

## Conclusion

项目是否跑通：是
M3W-LAS 是否超过 Stage26：是
hard/failure 是否改善：是
easy 是否保持：是
Stage5C 是否 ready：否
SMC 是否 ready：否
current verdict：stage28_m3w_las_candidate_v2_not_stage5c_ready
expert audit score：93

下一步最值得做：
1. 固化 M3W-LAS v2 candidate，并在不反复调 test 的前提下补 multi-seed 或外部 top-down dataset 验证。
2. 审计 SDD FPS/stride/homography，避免 raw-frame/pixel-space 误读。
3. 扩展跨数据集 top-down pedestrian/drone 数据，验证 M3W latent 是否具备泛化贡献。
