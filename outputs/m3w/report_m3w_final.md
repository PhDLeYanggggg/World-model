# M3W Final Report

- 项目名：M3W: Real-World Multimodal Agent-Scene World Model。
- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。
- SDD 是 pixel-space benchmark，不是 metric benchmark。
- t+50/t+100 是 raw annotation-frame horizon；effective seconds、homography、metric scale 未验证。
- self-audited / visual-prior labels 不是 human gold。
- Stage5C latent generative 未执行；SMC 未启用。
- execution backend: `torch_cpu_sequential`
- 本轮修复了 PyTorch OpenMP/SHM runtime：必须使用 sequential CPU 环境变量；MPS 在当前环境不可用。

- M3W variant: `hybrid`
- M3W t+50 improvement: `0.0797643165887999`
- M3W hard/failure improvement: `0.04591573857839959`
- M3W easy degradation: `0.008167665001088231`
- Stage26 t+50 improvement: `0.14583655843823773`
- beats Stage26 selector: `False`
- failure AUROC/AUPRC/ECE: `0.9346414494225567` / `0.5090746798518613` / `0.03996540148588829`
- full torch JEPA non-collapse: `False`

## Conclusion

M3W small pipeline 已真实执行，但不能称为 true 3D、foundation world model 或 latent generative world model。若未超过 Stage26 selector，当前 best deployable 仍是 Stage26 selector。
