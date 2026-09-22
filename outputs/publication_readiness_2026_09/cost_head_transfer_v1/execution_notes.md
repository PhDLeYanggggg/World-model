# Frozen Transfer: Execution and Reproduction

The completed readout follows registration `76cf3536`. All required experiment,
replay, arithmetic and forensic commands exited normally. No new model was fitted
and no threshold or normalization was changed. Use native arm64 `.venv-pytorch`;
the runner sets four compute threads, one inter-op thread and no loader workers.

```bash
.venv-pytorch/bin/python scripts/run_m3w_cost_head_transfer.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_cost_head_transfer.py
.venv-pytorch/bin/python scripts/run_m3w_cost_head_transfer.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_cost_head_transfer.py
.venv-pytorch/bin/python scripts/audit_m3w_cost_transfer_features.py
.venv-pytorch/bin/python scripts/audit_m3w_cost_transfer_support.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_cost_head_transfer.py tests/test_m3w_bounded_cost_head.py tests/test_m3w_native_nested.py tests/test_m3w_forecast_cost_bounds.py tests/test_m3w_cost_transfer_forensics.py
```

The first44-test run and subsequent10-test forensic run overlap on nine tests;
they represent45 distinct scoped tests, not54. The combined command above is
the narrow regression suite for this work. The full legacy suite is not claimed
here; some legacy tests have non-hermetic report-writing effects.

The immutable identity binds690 source/artifact files. Decision archives and
the completed decision manifest are frozen before transferred outcomes are
reduced. Private arrays remain under
`data/stage_cvpr2027_experiments/cost_head_transfer_v1/` and are not Git content.
The public analysis, replay, second implementation receipt and feature/support
diagnostics are lightweight evidence; no raw trajectories, checkpoints, images,
videos or third-party source are included.

Forensics was performed after the readout and is labeled diagnostic. It reads
past-only feature support; it does not update models or select a replacement
policy. Reproduction checks and known-byte hashes establish consistency of this
experiment, not independent statistical confirmation or population safety.
