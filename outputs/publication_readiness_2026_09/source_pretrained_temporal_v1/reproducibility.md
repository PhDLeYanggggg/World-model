# Reproduction

Registration was committed as `06f97771` before extraction or fitting.
Registration SHA256: `95823a77da228ea82d4ea39efdc1d067d9def86c3d7262c78acd0e7642edc5c6`.
Use the existing native arm64 `.venv-pytorch`; CPU four threads, workers zero.
The supplied local corpus, private parent checkpoints and official frozen encoder
weights must be available and match all registered hashes. Missing private assets
are an explicit prerequisite, not a successful reproduction. No downloads or fits
are hidden in the analyzer. Existing completed fits are verified and reused.

```bash
.venv-pytorch/bin/python scripts/run_m3w_source_pretrained_temporal.py --registration configs/m3w_source_pretrained_temporal_v1.json --phase prepare
.venv-pytorch/bin/python scripts/run_m3w_source_pretrained_temporal.py --registration configs/m3w_source_pretrained_temporal_v1.json --phase train
.venv-pytorch/bin/python scripts/run_m3w_source_pretrained_temporal.py --registration configs/m3w_source_pretrained_temporal_v1.json --phase replay
.venv-pytorch/bin/python scripts/analyze_m3w_source_pretrained_temporal.py --registration configs/m3w_source_pretrained_temporal_v1.json
.venv-pytorch/bin/python scripts/verify_m3w_source_pretrained_temporal.py --registration configs/m3w_source_pretrained_temporal_v1.json
.venv-pytorch/bin/python scripts/report_m3w_source_pretrained_temporal.py --registration configs/m3w_source_pretrained_temporal_v1.json
```

Interruptions recover from the last atomic 200-update checkpoint with optimizer,
sampler and Torch RNG state. The heartbeat records the current PID, trial and
step. Do not rerun completed fits under another identity or tune held results.
Images, image features, weights and checkpoints remain private and excluded from Git.
See `verification.json` for current artifact hashes and actual resume evidence.
