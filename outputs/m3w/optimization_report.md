# M3W Bounded Optimization Report

This report records the Stage27 local-small research loop after repairing the PyTorch runtime. It is not a full/medium result and it is not a CCF-A submission-ready claim.

## Runtime

- Root cause: Apple Silicon 上用 x86_64 Conda + Intel MKL/OpenMP 跑 PyTorch 训练，OpenMP 在训练并行初始化时触发 SHM 注册失败。
- Runtime fix:
  - use `.venv-pytorch/bin/python` arm64;
  - block macOS x86_64/Rosetta entry by default;
  - keep `num_workers=0`;
  - use `torch_threads=4` / `torch_interop_threads=2` for local-small.
- MPS is not available in the current arm64 environment.
- DataLoader multiprocessing remains disabled.

## Trials

| trial | backend | t+50 improvement | hard/failure improvement | easy degradation | failure AUROC | interaction AUROC | outcome |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| torch_small | torch_arm64_cpu_multithread | -0.0035411119 | -0.0081909895 | 0.0252436399 | 0.5180964703 | 0.5202319967 | true arm64 torch run; below baseline and easy fails |
| torch_small_optimized | torch_arm64_cpu_multithread | 0.0267446041 | 0.0240623951 | 0.0206217766 | 0.9191593613 | 0.7259047480 | bounded optimization; still below Stage26 and easy narrowly fails |
| torch_small_hard_focus | torch_arm64_cpu_multithread | 0.0288397670 | 0.0431480408 | 0.0040532351 | 0.9360968039 | 0.4447730028 | best Stage27 M3W run; easy preserved, hard/failure below gate |
| torch_medium | torch_arm64_cpu_multithread | 0.1308150291 | 0.1024016738 | 0.0106651783 | 0.9543455374 | 0.9976791527 | medium evidence run; hard/easy gates pass, but Stage26 remains stronger |

## Best Deployable Model

Stage26 remains the best deployable model:

- t+50 improvement: `0.14583655843823773`
- hard/failure improvement: `0.11234058960663984`
- easy degradation: `0.01808836280803794`

## Conclusion

M3W is now a real executable research track with the correct arm64 PyTorch runtime and a medium evidence run. It is still not yet a CCF-A submission candidate because the Stage26 feature-complete cost-aware selector remains stronger on t+50 and hard/failure, JEPA latent non-collapse remains below gate, and scene/goal lift is not proven. The shortest path is to use M3W latent/Transformer features as auxiliary Stage26 selector features and run retrained ablations plus multi-seed variance.
