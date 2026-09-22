# Execution and Reproduction

2026-09-22, native Apple arm64 `.venv-pytorch`, sklearn 1.8.0. Frozen registration
commit `1985a3bc` preceded fitting. CPU fitting threads 4, deterministic prediction
threads 1, DataLoader/worker processes 0. No MPS fallback or new Torch training
is claimed for these tree regressors. Frozen upstream neural predictions are
reused after lineage/hash checks. CREATE path/auth remains unresolved; this
local run did not require HPC and makes no claim about remote jobs or assets.

## Commands

```sh
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_forest_cost_head.py tests/test_m3w_temporal_fit_support.py tests/test_m3w_temporal_intervention.py
.venv-pytorch/bin/python scripts/run_m3w_forest_cost.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_forest_cost.py --view coupa_seed17 --arm ramp --stop-at 16
.venv-pytorch/bin/python scripts/run_m3w_forest_cost.py --resume
.venv-pytorch/bin/python scripts/run_m3w_forest_cost.py --evaluate
.venv-pytorch/bin/python scripts/run_m3w_forest_cost.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_forest_cost.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_forest_verifier.py
.venv-pytorch/bin/python scripts/audit_m3w_forest_fit.py
```

The first six head tests train small synthetic forests and verify exact recovery,
not model quality on SDD. The combined 24 tests include prior unchanged target/
support tests; two additional verifier tests pass. Do not call this the full
legacy suite. Repeated `--resume` validates completed receipts instead of fitting
again. It never changes a frozen endpoint. `--verify` replays scores and the
entire readout, not training. Independent arithmetic checks are a separate file.

The preflight bound 1,249 dependencies. PID 23059 completed the 16-tree full-row
pilot in 3.34 fitting seconds; PID 23130 resumed the matrix to all 24 endpoints.
A live read-only process sample showed about 3.8 GiB RSS and four busy compute
threads. The complete matrix has 3,072 trees, 619.66 recorded fitting seconds,
and about 404 MiB of checkpoints. Evaluation PID 24232 and replay PID 24328
returned exit0. Separate arithmetic verification also completed with exit0.
The fitting/held diagnostic completed and then reproduced its immutable JSON
exactly in a second execution. All required experiment sessions are terminal;
there is no ongoing training hidden behind this result report.

Private path: `data/stage_cvpr2027_experiments/forest_cost_v1/`.
Atomic checkpoints are `trials/<site>_seed<seed>/<arm>/checkpoint.joblib`.
`heartbeat.json`, `events.jsonl`, `identity.json`, completion receipts and
row-level decision archives remain local. They are excluded by the existing
`data/stage*/` ignore rule. Do not commit model binaries or data. If interrupted,
verify the known PID/session first, then use `--resume` after it is terminal;
the last saved batch is retained and no old checkpoint is deleted.

The final report separates repeated checkpoint reproduction, independently
implemented arithmetic, and actual independent scientific confirmation. The
last of these is **not_run**. No risk calibration, deployment, latent generation
or SMC was executed.
