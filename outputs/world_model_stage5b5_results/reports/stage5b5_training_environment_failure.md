# Stage 5B.5 Training Environment Failure / Recovery Note

PyTorch/OMP training first failed with `OMP: Error #179: Function Can't open SHM failed`. The next run uses single-thread environment variables and a fallback path if needed.

Historical process note: PID 48952 remained in `UEs` uninterruptible state after targeted SIGKILL. This was an OS/runtime OpenMP shared-memory hang, not a Python exception.

Updated status after environment restart: the cleaned `.venv_m3_torch` run completed PyTorch deterministic temporal-interaction training and produced three checkpoints:

- `outputs/checkpoints/stage5b5/temporal_interaction_direct_multi_horizon.pt`
- `outputs/checkpoints/stage5b5/temporal_interaction_recurrent_rollout.pt`
- `outputs/checkpoints/stage5b5/temporal_interaction_hybrid.pt`

This resolves the runtime blocker. It does not resolve the modeling gate: PyTorch results still fail the deterministic learned dynamics gate overall, so Stage 5C latent generative modeling and SMC remain disabled.
