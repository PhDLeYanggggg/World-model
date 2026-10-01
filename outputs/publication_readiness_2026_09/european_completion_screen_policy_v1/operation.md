# Execution and Reproduction

Registered contrast: `9650a307`.
Source choices frozen: `97b4f8f2`.
All216 causal action views frozen before outcomes: `466bcb18`.

From the repository root with private input/checkpoint assets present:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_completion_screen_policy.py replay_select
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_completion_screen_policy.py replay_evaluate
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m pytest tests/test_m3w_completion_screen_policy.py tests/test_m3w_unknown_outcome_bounds.py tests/test_m3w_source_policy_selection.py -q
```

Initial sequence was register, commit, select, replay_select, commit, decide,
commit, evaluate, replay_evaluate, report. Model/feature/parent-seal checks occur
before every phase. Immutable output comparison prevents changing a result under
the same registration. Source choices are reconstructed from sealed source-only
diagnostic aggregates, not transfer outcomes.

The evaluation rebuilds both policy actions, verifies original parent action
hashes, checks same per-query counts, independently recalculates native metrics
and scalar completion bounds, then writes aggregate JSON only. Replay includes
all action and numeric readout checks. Readout labels never enter the causal
decision API; completion diagnostics are not deployment eligibility masks.

Native arm64, CPU4/inter-op1/workers0, no new gradient updates, row cache or
checkpoint. Atomic immutable receipts and a process lock protect completed work;
heartbeat/event records live in the ignored private directory. Planned output
allowance16MiB. Earlier training disk-reserve guards remain unchanged.

Full legacy test suite and new independent confirmation: not_run. These scoped
checks establish arithmetic/replay properties, not the research hypothesis.
CREATE job37602475 remains a separate, unverified remote replication; a timeout
does not justify cancelling, restarting or duplicating it. No new HPC job.
