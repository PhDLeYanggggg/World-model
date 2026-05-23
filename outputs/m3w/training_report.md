# M3W Training Report

- This run used the CPU-safe NumPy backend because local PyTorch/OpenMP SHM execution blocked before heartbeat.
- The PyTorch JEPA/Transformer implementation is present, but this checkpoint is not a full torch JEPA-Transformer success.
- No Stage5C latent generative execution, no SMC, no ordinary residual training.
- backend: `numpy_safe_fallback_due_torch_openmp_shm_blocker`
- best variant: `hybrid`
- best checkpoint: `outputs/m3w/checkpoints/best_small.pt`
