# Reproduction

Registration committed as 4b5dadd9 before fitting. SHA256: `90f8ece637af299f7b89f004abd77bc548895f0b469096b4fd7ed5a528439461`.
Requires hash-matching private parent data, feature store, old controls and official
image encoder weights. Missing assets are prerequisites, not completed reproduction.

```bash
.venv-pytorch/bin/python scripts/run_m3w_source_temporal_centered.py --registration configs/m3w_source_temporal_centered_v1.json
.venv-pytorch/bin/python scripts/run_m3w_source_temporal_centered.py --registration configs/m3w_source_temporal_centered_v1.json --replay
.venv-pytorch/bin/python scripts/analyze_m3w_source_temporal_centered.py --registration configs/m3w_source_temporal_centered_v1.json
.venv-pytorch/bin/python scripts/verify_m3w_source_temporal_centered.py --registration configs/m3w_source_temporal_centered_v1.json
.venv-pytorch/bin/python scripts/report_m3w_source_temporal_centered.py --registration configs/m3w_source_temporal_centered_v1.json
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_source_temporal_information.py tests/test_m3w_source_temporal_centered.py tests/test_m3w_source_pretrained_temporal.py
```

200-update atomic checkpoints contain optimizer, Torch RNG, sampler RNG and draw
counts. Reuse the same entry after interruption. Complete trials hash-verify and
skip training. Heartbeat records PID and step. No outcome-dependent restart.
The full legacy integration/training suite is not rerun for this scoped repair.
Past-only input audit entry: `scripts/audit_m3w_source_temporal_information.py`.
No alternative input or target columns are inferred from a filename.
