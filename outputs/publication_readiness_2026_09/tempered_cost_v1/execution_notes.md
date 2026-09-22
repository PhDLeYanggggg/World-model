# Execution and Reproduction

Native arm64 `.venv-pytorch`, CPU4/inter-op1/workers0. No resource probing,
DataLoader multiprocessing, architecture fallback or new CREATE job. Local
head-only training fits available resources;55GiBwas free before the study.
No current remote scheduler state is claimed.

Registration5a9d02ab was pushed before fitting. Config/code/tests/design are
identity-bound and unchanged.904bindings pass preflight. Single-process lock,
atomic500-step checkpoints,100-step PID/progress events. Decisions are archived
before aggregate outcome reduction; held labels are not inference arguments.
Training supervision and lineage checks are separate permitted label uses.

| Process | Session / PID if recorded | Terminal result |
|---|---|---|
|Preflight|49494 /79599|exit0|
|100-update pilot|77677 /79855|exit0; checkpoint resumed|
|All12fits|39504 /79902|exit0;36,000total updates|
|Fixed readout|55742 /80046|exit0; primary criterion fails|
|Full score replay|28315 /80121|exit0; identical analysis|
|Separate arithmetic|79607|exit0|
|99scoped tests|43409|exit0|
|Post-readout fit/selection audit|62432|exit0|

Each head has22,978parameters;9,216,000draws, no unknown-supervision draw.
Head fitting28.07731seconds excludes assembly/hashing/readout/verification and
the earlier7.75hours of producer fitting. The budget was not shrunk; cached
small cost heads are inexpensive. No NumPy fallback.

The first-run sequence, with existing source assets, was:

```bash
.venv-pytorch/bin/python scripts/run_m3w_tempered_cost.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_tempered_cost.py --view coupa_seed17 --stop-at 100
.venv-pytorch/bin/python scripts/run_m3w_tempered_cost.py --resume
.venv-pytorch/bin/python scripts/run_m3w_tempered_cost.py --evaluate
.venv-pytorch/bin/python scripts/run_m3w_tempered_cost.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_tempered_cost.py
.venv-pytorch/bin/python scripts/audit_m3w_tempered_cost.py
```

For these completed artifacts use verify, not the pilot. A fresh retraining
requires a separate identified output/config without overwriting this frozen
run. For interruption, first verify the original process ended, then resume
its last atomic checkpoint; a polling timeout is not permission to restart.
The independent-verification filename denotes separate arithmetic by the
same agent, not an independent team or final test.

All heavy checkpoints and feature/prediction caches remain local. Git receives
source, config, reports and light metrics only. No original role reassignment,
threshold search, model selection, deployment, Stage5C or SMC.
