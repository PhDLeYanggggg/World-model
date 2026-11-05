# Reproduce The Direct Tail Pilot

Use native arm64 `.venv-pytorch/bin/python`, CPU4/interop1/workers0.
The registration binds model, runner, config, protocol and tests before fitting.
The two losses use identical initialization, preprocessing, sampling, proposal
support and update count. No future outcome is supplied to inference.

```sh
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_direct_tail_risk.py tests/test_m3w_proposal_conditioned_risk.py tests/test_m3w_proposal_risk_runner.py
.venv-pytorch/bin/python scripts/run_m3w_moving_direct_tail_risk.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_direct_tail_risk.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_direct_tail_risk.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_direct_tail_risk.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_direct_tail_risk.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_direct_tail_risk.py
```

Use `--resume` only for an interrupted, incomplete phase. Completed phase
receipts are immutable and should not be removed to rerun them. Existing
checkpoints can be rechecked by the report audit; a fresh independent rerun
needs a new explicit output namespace, not overwriting these results.
The private SOURCE assets, forecast bank and pretrained parent heads are local
dependencies, not bundled with the public repository. Git alone is insufficient
to reproduce those data. Checkpoints and caches are intentionally not committed.
The complete historical test suite was not rerun for this scoped model change.
