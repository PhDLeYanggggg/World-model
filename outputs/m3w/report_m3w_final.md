# M3W Final Report

- 项目名：M3W: Real-World Multimodal Agent-Scene World Model。
- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。
- SDD 是 pixel-space benchmark，不是 metric benchmark。
- t+50/t+100 是 raw annotation-frame horizon；effective seconds、homography、metric scale 未验证。
- self-audited / visual-prior labels 不是 human gold。
- Stage5C latent generative 未执行；SMC 未启用。
- execution backend: `torch_arm64_cpu_multithread`
- Runtime root cause: Apple Silicon 上默认 x86_64 Conda + Intel MKL/OpenMP 会触发 SHM 注册失败；训练必须使用 `.venv-pytorch/bin/python` arm64，`num_workers=0`，torch threads 4/8。

- M3W variant: `hybrid`
- M3W t+50 improvement: `0.1308150291442871`
- M3W hard/failure improvement: `0.10240167379379272`
- M3W easy degradation: `0.010665178298950195`
- Stage26 t+50 improvement: `0.14583655843823773`
- beats Stage26 selector: `False`
- failure AUROC/AUPRC/ECE: `0.9543455373640822` / `0.6381205770048155` / `0.014374548431523144`
- full torch JEPA non-collapse: `False`

## Conclusion

M3W Stage27 evidence sprint 已执行，但不能称为 true 3D、foundation world model、latent generative world model 或 CCF-A candidate。若未超过 Stage26 selector，当前 best deployable 仍是 Stage26 selector。
