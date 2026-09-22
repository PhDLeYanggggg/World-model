# Capacity/Duration Execution and Recovery

Registration `621a6eb2` was committed and pushed before the real pilot. The
config, runner, model, evaluator, verifier, protocol tests and registration are
identity-bound and unchanged. Native arm64 `.venv-pytorch`, CPU4/inter-op1,
num_workers0; no multiprocessing/resource probe/NumPy fallback/new CREATE job.
Local resources suffice for cached cost heads;54GiBfree was verified at readout.
No current CREATE scheduler state is claimed.

| Phase | Session / PID if recorded | Terminal result |
|---|---|---|
|Preflight|72678 /82334|exit0;942 dependency bindings|
|Wide coupa17 pilot100|67917 /82499|exit0; same checkpoint resumed|
|12wide paths and12narrow continuations|77942 /82560|exit0;252000 new updates|
|Fixed readout|18198 /83650|exit0; registered primary empirical checks pass|
|Full endpoint replay|24804 /83766|exit0; identical analysis|
|Separate arithmetic|25115|exit0;144choices/1152reductions|
|Post-readout same-population audit|14646|exit0;960 diagnostic records|
|50 scoped tests|18137|exit0|

The100-step pilot is part of the full wide path. Narrow checkpoints copy the
exact3000-step parent model, optimizer, RNG and sample counts into new output
folders before continuation; parents are not changed. Twelve wide3000 endpoints
are immutable prefixes, not extra independent fits.36 new endpoints plus12 old
endpoints are replayed. All first3000 wide draws match the old narrow draws;
all12000 narrow/wide draws match. No incomplete-supervision row is sampled.

New compute:252000 updates,64512000 draws,247.80623 recorded fit seconds. Old
narrow compute36000 updates/9216000 draws and upstream7.75hour producer fitting
are not included in that new total. Timings exclude loading and verification.

The initial sequence was:

```bash
.venv-pytorch/bin/python scripts/run_m3w_cost_capacity.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_cost_capacity.py --view coupa_seed17 --width wide --stop-at 100
.venv-pytorch/bin/python scripts/run_m3w_cost_capacity.py --resume
.venv-pytorch/bin/python scripts/run_m3w_cost_capacity.py --evaluate
.venv-pytorch/bin/python scripts/run_m3w_cost_capacity.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_cost_capacity.py
.venv-pytorch/bin/python scripts/audit_m3w_cost_capacity.py
```

For completed artifacts use verify and the arithmetic verifier, not a fresh
pilot. New training needs a separately identified output contract. On an actual
interruption first verify that the original process ended, then resume the last
atomic checkpoint. Polling timeouts alone do not justify a restart. Checkpoints
are saved every500updates and PID/loss heartbeat every100.

Decisions are archived before aggregate outcome readout. Labels remain loss/eval
inputs only. No new normalization, goal construction, threshold search or role
opening. Unknown and incomplete futures are reported, not safe zero errors.
The separate verification file denotes a second arithmetic implementation by
the same agent, not independent confirmation.

Only code/tests/reports/light metrics go to Git. Data, forecasts, features,
checkpoints, videos/images, third-party assets and the environment remain local.
No deployment, Stage5C or SMC.
