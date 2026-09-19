# Reproduction

Registration SHA256: `4a1974e1ca3890efa9fcdd91fa430d31fc2194cfbade9ecc7e860aad2f1eddca`. Private hash-matching
source data, past image features, episode mapping and cached controls are needed.
Missing private assets are not a completed reproduction. Keep bound files frozen.

```bash
.venv-pytorch/bin/python scripts/run_m3w_source_importance_sampling.py --registration configs/m3w_source_importance_sampling_v1.json --check-objective
.venv-pytorch/bin/python scripts/run_m3w_source_importance_sampling.py --registration configs/m3w_source_importance_sampling_v1.json
.venv-pytorch/bin/python scripts/run_m3w_source_importance_sampling.py --registration configs/m3w_source_importance_sampling_v1.json --replay
.venv-pytorch/bin/python scripts/analyze_m3w_source_importance_sampling.py --registration configs/m3w_source_importance_sampling_v1.json
.venv-pytorch/bin/python scripts/verify_m3w_source_importance_sampling.py --registration configs/m3w_source_importance_sampling_v1.json
.venv-pytorch/bin/python scripts/report_m3w_source_importance_sampling.py --registration configs/m3w_source_importance_sampling_v1.json
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_source_importance_sampling.py tests/test_m3w_source_episode_sampler.py
```

The included pilot uses `--trial coupa_geometry_seed17 --stop-at 100` before
the full run. Interrupted runs resume from 200-step atomic checkpoints; never
restart a live process after an observation timeout. PID/heartbeat/training log
and all 24 checkpoints remain local. Completed resume performs zero updates.
No CUDA/MPS resource probing or multiprocessing. On Darwin reject non-arm64 before
Torch import. Training cost does not extrapolate to end-to-end encoder fitting.
The full legacy test suite can rewrite old artifacts and was not rerun here.
