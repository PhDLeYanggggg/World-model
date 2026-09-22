# Execution and Recovery

Registration `06f9a635` was committed and pushed before training. Frozen config,
runner/evaluator/verifier/tests and registration remain unchanged. Nativearm64
Torch CPU4/inter-op1/workers0; no NumPy substitute or new CREATE job. Local
cached-head training is feasible. No current remote scheduler state is claimed.

| Phase | Session / PID where recorded | Terminal result |
|---|---|---|
|Preflight|12606 /85067|exit0,1023 dependency bindings|
|Native coupa17 pilot100|44960 /85220|exit0, resumed rather than restarted|
|All24 new fits|86017 /85257|exit0,288000updates|
|Fixed readout|57227 /85817|exit0, primary joint check false|
|Full36checkpoint replay|62378 /85925|exit0, identical analysis|
|Separate arithmetic|76581|exit0,108choices/864reductions|
|48 scoped tests|85999|exit0,1.81seconds|

New head-fit compute is245.01177seconds, excluding loading and verification.
New73728000draws/288000updates include the100-update pilot. Twelve frozen
intermediate references have separate earlier144000updates/36864000draws.
Neither those heads nor upstream trajectory predictors were refitted here.

Initial execution sequence, not a request to restart completed fits:

```bash
.venv-pytorch/bin/python scripts/run_m3w_cost_budget_matched.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_cost_budget_matched.py --view coupa_seed17 --arm native --stop-at 100
.venv-pytorch/bin/python scripts/run_m3w_cost_budget_matched.py --resume
.venv-pytorch/bin/python scripts/run_m3w_cost_budget_matched.py --evaluate
.venv-pytorch/bin/python scripts/run_m3w_cost_budget_matched.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_cost_budget_matched.py
```

For completed artifacts use the two verification commands. A genuinely
interrupted fit can resume only after checking the original process ended;
a polling timeout is not a failure. Atomic checkpoints every500updates and
PID/loss/gradient heartbeats every100updates support recovery. Single-process
locking prevents duplicate training. Pilot/new continuation counts must not
double-count inherited steps.

The separate verifier reconstructs labels, policy bits, forecast errors and reductions
with different arithmetic, but shares source loading and preprocessing helpers.
It is same-agent verification, not independent research replication. The
792 fitting diagnostics are replayed by the main runner, not independently
implemented. All36endpoint scores and cached reference summaries match exactly.

The48tests cover budget/schema/sampler matching, tiny real Torch training,
resume equivalence, objective behavior, policy inference exclusion, contrasts
and diagnostic arithmetic. The full legacy suite was not rerun: its known
nonhermetic integration tests can rewrite unrelated reports/data state.

Only reports and light aggregate metrics accompany this result commit. No
raw inputs, forecasts, features, caches, checkpoints, images/videos, third-party
assets or environment are committed. Deployment/Stage5C/SMC remain disabled.
