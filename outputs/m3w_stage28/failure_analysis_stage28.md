# Stage28 Failure Analysis

- Stage26 的优势来自 cost-aware expected-FDE policy + conservative fallback；它直接优化基线选择损失。
- M3W latent 若没有超过 Stage26，主要说明当前 JEPA/Transformer hidden features 未提供足够可迁移的 selector signal。
- JEPA non-collapse 不等于 downstream lift；Stage28 只把 downstream metric 作为贡献证据。
- Hybrid 不如 Stage26 时不得部署 Hybrid，只能作为辅助 diagnostics。
- Stage28 当前结果显示 all-latent selector 有增益；但 no-scene ablation 没有下降，因此 scene-only contribution 仍不能作为强主 claim。

- best Stage28 t+50: `0.1686288243790961`
- Stage26 t+50: `0.14583655843823773`
- best Stage28 hard/failure: `0.1336398986813968`
- Stage26 hard/failure: `0.11234058960663984`
