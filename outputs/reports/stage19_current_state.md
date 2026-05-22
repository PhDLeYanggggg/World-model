# Stage 19 Current State

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D per-agent multi-agent trajectory world-state scaffold。
- BPSG-MA v1 可运行，但部署策略仍是 strongest causal baseline fallback + failure diagnostics。
- Stage 17 selector 有部分提升，但 hard/failure 与 correction specialist 仍不过 gate。
- Stage 18 JEPA non-collapse，但 downstream heads 没有提升。
- Stage 5C latent generative 仍不 ready。
- SMC 仍不 ready。

为什么 Stage18 JEPA 没有 downstream lift：
- Stage18 used derived preview/raster context rather than raw scene/video data.
- Hard/failure and official long-horizon rows remain limited.
- JEPA probes did not improve failure AUROC, selector, or correction metrics.

为什么现在补 WAM-style data：The next bottleneck is data diversity and controllable hard/failure supervision, not another residual head.
为什么 simulation data 只能做 pretraining/stress test：pretraining and stress test only; not real-world success.
为什么 human/egocentric video 只能做 representation pretraining：representation pretraining only; not top-down trajectory ground truth.
为什么 official benchmark 仍必须依赖真实 top-down trajectory：real top-down pedestrian/drone trajectories only.
当前 latent generative 和 SMC 为什么仍禁止：deterministic correction, official horizon, and hard/failure gates remain insufficient.
