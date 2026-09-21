# Protected-Risk Execution and Reproduction

Registration/code/config commit `8a74fff5` was pushed before training. Bound
files must remain unchanged; use a new experiment version for modifications.
Private artifacts: `data/stage_cvpr2027_experiments/native_protected_risk_v1/`.
No raw data, row cache or checkpoint is included in the Git result commit.

## Completed Commands

```sh
.venv-pytorch/bin/python scripts/run_m3w_native_protected_risk.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_native_protected_risk.py --view coupa_seed17 --arm all_harm --stop-at 100
.venv-pytorch/bin/python scripts/run_m3w_native_protected_risk.py --resume
.venv-pytorch/bin/python scripts/run_m3w_native_protected_risk.py --evaluate
.venv-pytorch/bin/python scripts/run_m3w_native_protected_risk.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_native_protected_risk.py
```

The pilot contributes 100 updates, resumed training 71,900; total 72,000.
Pilot PID74463/session81226, training PID74520/session93936, evaluation
PID74763/session42690, replay PID74841/session29123 all exit0.
Independent verifier session3443 exits0 after numerical-tolerance repairs
described in failure_analysis.md. Do not restart these completed PIDs.

Native arm64 Torch CPU4, inter-op1, DataLoader workers0. No CREATE job or remote
queue state was checked/claimed. Fitting takes 56.279 summed seconds on cached
357-dimensional inputs, excluding prior forecast training and materialization.
Every head has 23,042 parameters. Checkpoints every500updates, heartbeat100.
Full identity, RNG, sampler exposure, optimizer and preprocessing are preserved.
Completed `--resume` verifies cached endpoints rather than fitting new models.

## Tests and Checks

```sh
.venv-pytorch/bin/python -m pytest tests/test_m3w_native_protected_risk.py tests/test_m3w_native_protected_risk_verifier.py tests/test_m3w_native_gain_harm.py tests/test_m3w_native_gain_harm_verifier.py tests/test_m3w_native_matched_coverage.py tests/test_m3w_native_metrics.py tests/test_m3w_native_forecast.py tests/test_m3w_native_forecast_verification.py tests/test_m3w_native_nested.py tests/test_m3w_native_cost_views.py tests/test_m3w_experiment_contract.py
```

116 related tests pass in2.42s, session89855. This is not a new run of the entire
historical repository suite. Earlier registered implementation tests remain
unchanged; the independent verifier has three separate regression tests.

Replay: 24heads, 12rebuilt normalizers, 12paired draw checks, 1,054,536 score rows.
Independent: 3,163,608 repeated target rows, 132selection hashes, 792scene
reductions, 48event metrics. Query/seed repetitions are not independent samples.

Analysis SHA256:
`3d6785b3558cebd945d4115787bd2b88f55059d7a04eb5d240c6bebe2380bc7c`.
Replay SHA256:
`b15b52b74c99343ef953289ed334cde4ca52822ccfa9e669492bf8cbf97c0806`.
Independent verification binds its own implementation and shared arithmetic
helpers. Frozen modeling/evaluation files were not changed to satisfy checks.

No source threshold search, lucky-seed choice, independent risk calibration or
closed-role readout. This is a completed negative safety experiment, not a
completed publication goal. All four source scenes remain design-exposed.
