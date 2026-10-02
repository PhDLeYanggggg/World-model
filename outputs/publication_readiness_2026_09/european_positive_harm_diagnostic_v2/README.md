# Reproduction

Registered commit:59f41b52. Native arm64 Python,4 compute threads,0 workers.
Checkpoint files remain under the existing owned CREATE M3W directory; no
weights, raw data or numerical caches are in this public evidence package.

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv-pytorch/bin/python scripts/run_m3w_positive_harm_diagnostic_v2.py verify
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/verify_m3w_positive_harm_diagnostic_v2.py
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m pytest tests/test_m3w_positive_harm_diagnostic.py tests/test_m3w_positive_harm_diagnostic_v2.py -q
```

Use`run --resume` only for an interrupted incomplete run; `verify` replays a
completed run without overwriting its immutable reports. Eight scoped tests
passed. The independent verifier uses scalar sums and a separate bootstrap
reduction, not the fitted-model evaluator. See conclusions.md for the real
precision cause and correction to the original amendment description.
