# M3W Final Report

- 项目名：M3W: Real-World Multimodal Agent-Scene World Model。
- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。
- SDD 是 pixel-space benchmark，不是 metric benchmark。
- t+50/t+100 是 raw annotation-frame horizon；effective seconds、homography、metric scale 未验证。
- self-audited / visual-prior labels 不是 human gold。
- Stage5C latent generative 未执行；SMC 未启用。
- execution backend: `numpy_safe_fallback_due_torch_openmp_shm_blocker`
- 本轮 PyTorch local-small execution 被本机 OpenMP/SHM 卡死；当前 checkpoint 是 CPU-safe NumPy fallback，不是完整 torch JEPA-Transformer 成功。

- M3W variant: `hybrid`
- M3W t+50 improvement: `-0.0003516511641914466`
- M3W hard/failure improvement: `0.00565237402143326`
- M3W easy degradation: `0.006164539707586902`
- Stage26 t+50 improvement: `0.14583655843823773`
- beats Stage26 selector: `False`
- failure AUROC/AUPRC/ECE: `0.9364380811532312` / `0.5470499708612074` / `0.4459426108889468`
- diagnostic latent non-collapse: `True` (NumPy fallback only; full JEPA gate remains false)

## Conclusion

M3W small pipeline 已真实执行，但不能称为 true 3D、foundation world model 或 latent generative world model。若未超过 Stage26 selector，当前 best deployable 仍是 Stage26 selector。
