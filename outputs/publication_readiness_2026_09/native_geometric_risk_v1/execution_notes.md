# Geometric-Risk Reproduction

Registration `91581b9a` was pushed before pilot/fulltraining. Frozen configs,
model, runner, tests and registration must not change under the existing private
identity. Start a new version for scientific/code changes. No new CREATE job was
needed or remote scheduler state claimed: the local arm64 CPU path was measured
with real training. CPU4/inter-op1/workers0; about60GiB disk free at preflight.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_native_conditional_support.py
.venv-pytorch/bin/python scripts/run_m3w_native_geometric_risk.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_native_geometric_risk.py --view coupa_seed17 --arm base_event --stop-at 100
.venv-pytorch/bin/python scripts/run_m3w_native_geometric_risk.py --resume
.venv-pytorch/bin/python scripts/run_m3w_native_geometric_risk.py --evaluate
.venv-pytorch/bin/python scripts/run_m3w_native_geometric_risk.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_native_geometric_risk.py
```

Pilot session82692/PID77432, full98631/PID77587, evaluation90557/PID77951,
replay60294/PID78069, independent verifier42191 all exit0. Support audit21135
and preflight13478 exit0. Do not restart completed PIDs. Completed resume checks
endpoints without adding updates. The pilot's100updates plus143,900resumed
updates equal144,000; total36,864,000draws, no unknown-risk training rows.
48heads have23,233parameters each. Summed fitting115.896seconds excludes
feature loading/verification, prior forecaster training and previous experiments.

Private checkpoints, row scores, heartbeat/events and identities are under
`data/stage_cvpr2027_experiments/native_geometric_risk_v1/`. Atomic checkpoints
every500updates include RNG, optimizer, preprocessing and draw counts; heartbeat
every100updates. On interruption resume the same registered run, not a new
scientific configuration. Existing complete receipts and source hashes are checked.

```sh
.venv-pytorch/bin/python -m pytest tests/test_m3w_native_geometric_risk.py tests/test_m3w_native_conditional_support.py tests/test_m3w_native_protected_risk.py tests/test_m3w_native_protected_risk_verifier.py tests/test_m3w_native_gain_harm.py tests/test_m3w_native_gain_harm_verifier.py tests/test_m3w_native_matched_coverage.py tests/test_m3w_native_metrics.py tests/test_m3w_native_forecast.py tests/test_m3w_native_forecast_verification.py tests/test_m3w_native_nested.py tests/test_m3w_native_cost_views.py tests/test_m3w_experiment_contract.py
```

123relatedtests pass3.04s in session96632. The full historical repository suite
was not rerun. Independent verifier also compiles and completes on real artifacts.
424dependencybindings,48checkpointreplays/2,109,072probability rows,36paired
draw checks; separate240choice hashes/1,440scene reductions/6,327,216repeated
target rows/96event metric checks. Maximum ECE arithmetic difference1.755e-9
uses the existing verifier's dtype-rounding tolerance, not a scientific margin.

Analysis SHA256:
`479d3d68717f545f0f73e00ad26fb5fb73ffc79600a2e6ce5961be34d85a1f60`.
Replay SHA256:
`f66ce2eec5ae2c546d81083729376514fb8db87f5beeb5f1cadf2e3243dd7044`.
The independent verifier binds its own implementation and arithmetic helpers.
All reports retain failed arms; no held threshold selection or deployment.
