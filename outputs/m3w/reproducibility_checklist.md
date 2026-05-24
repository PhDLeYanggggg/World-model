# M3W Reproducibility Checklist

- [x] Use `.venv-pytorch/bin/python` arm64.
- [x] Refuse macOS x86_64/Rosetta torch training by default.
- [x] DataLoader multiprocessing disabled (`num_workers=0`).
- [x] Checkpoints saved under `outputs/m3w/checkpoints/`.
- [x] No future endpoint input.
- [x] No central velocity official input.
- [x] No test endpoint goal construction.
- [x] Stage5C not executed.
- [x] SMC not enabled.
- [ ] Multi-seed variance not yet complete.
- [ ] Retrained ablations not yet complete.
