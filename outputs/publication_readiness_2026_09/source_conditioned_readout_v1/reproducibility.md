# Reproduction and Recovery

Registration SHA256: `ffe3cbe194cc8a86509dafdc96169944df7eabb2c706a01fb98eb13f5d666ed4`. Original hash-matching
source data, embeddings, event groups and controls must be present locally.
Missing private assets are not a completed reproduction.

```bash
.venv-pytorch/bin/python scripts/run_m3w_source_conditioned_readout.py --registration configs/m3w_source_conditioned_readout_v1.json
.venv-pytorch/bin/python scripts/run_m3w_source_conditioned_readout.py --registration configs/m3w_source_conditioned_readout_v1.json --replay
.venv-pytorch/bin/python scripts/analyze_m3w_source_conditioned_readout.py --registration configs/m3w_source_conditioned_readout_v1.json
.venv-pytorch/bin/python scripts/verify_m3w_source_conditioned_readout.py --registration configs/m3w_source_conditioned_readout_v1.json
.venv-pytorch/bin/python scripts/report_m3w_source_conditioned_readout.py --registration configs/m3w_source_conditioned_readout_v1.json
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_source_gradient_diagnostic.py tests/test_m3w_source_conditioned_readout.py tests/test_m3w_source_conditioned_resume.py tests/test_m3w_source_importance_sampling.py tests/test_m3w_source_episode_sampler.py tests/test_m3w_source_pretrained_temporal.py tests/test_m3w_source_temporal_centered.py
```

The included pilot uses `--trial coupa_geometry_seed17 --stop-at 100`. Full
execution resumes that checkpoint. Checkpoints every200steps retain model,
optimizer, RNGs, sampler, readout gain and identity. Inspect PID/heartbeat/log
before restarting; an observation timeout is not a stopped process. A completed
run must add zero updates and leave artifacts unchanged. Native arm64 CPU4,
inter-op1, workers0, no resource probing or DataLoader multiprocessing.

The gradient audit can be recomputed separately with
`scripts/audit_m3w_source_gradients.py --registration configs/m3w_source_gradient_diagnostic_v1.json --replay`.
Its gradients concern only predecessor training rows; it does not score a new
predictor. Large private vectors, caches, data and checkpoints are not in Git.
