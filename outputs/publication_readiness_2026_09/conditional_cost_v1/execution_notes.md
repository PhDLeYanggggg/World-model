# Execution and Reproduction

Registration `623e02b0` was committed and pushed before real training. The
preflight validated1,074 dependency bindings;39 focused tests passed. Native
arm64 `.venv-pytorch`, Torch2.12.0, CPU4/inter-op1, workers0 were used. No new
CREATE job was submitted; this is not a current remote scheduler audit.

First execution, preserved for reproducibility rather than rerunning completed
fits:

```bash
.venv-pytorch/bin/python scripts/run_m3w_conditional_cost.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_conditional_cost.py --view coupa_seed17 --stop-at 100
.venv-pytorch/bin/python scripts/run_m3w_conditional_cost.py --resume
.venv-pytorch/bin/python scripts/run_m3w_conditional_cost.py --evaluate
```

Pilot session83996/PID88200 and resumed full session50713/PID88275 exited0.
All12 heads completed12,000updates each. The resumed100updates are included once
in144,000updates and36,864,000draws. Summed recorded head-fit time is133.80316s;
this excludes predictor fitting, cache loading and checks. All unknown-supervision
draw counts are zero. Old36 heads are verified references, not fresh fits.

Readout session62078/PID88973 exited0. Full replay session99054/PID89100 exited0.
The new checkpoint forward scores and complete analysis match, including all
72 conditional diagnostic records. Existing controls retain their earlier
hash-verified replay; this run does not claim48fresh checkpoint replays.

For completed artifacts, use verification instead of repeating the pilot:

```bash
.venv-pytorch/bin/python scripts/run_m3w_conditional_cost.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_conditional_cost.py
```

The separate verifier shares preprocessing and model forward with the main
implementation but recalculates supervision, weights, decisions, error
reductions and paired contrasts. The conditional diagnostic table is replayed,
not independently reimplemented. See `replay.json` and
`independent_verification.json` for exact checked counts and analysis binding.
Separate verifier session91177 exited0:36policy choices,288scene reductions,
1,581,804supervision rows and12weight vectors agree. No required experiment
process remains running at this record's completion.

Checkpoints, identity, history and score archives remain under ignored
`data/stage_cvpr2027_experiments/conditional_cost_v1/`. Only code, config, reports
and light aggregate receipts belong in Git. Never start another writer while
the runner lock/process is live. On interruption, confirm process termination
then resume the last atomic checkpoint. Do not change hash-bound source files.

No original closed role was opened. No calibration, deployment, Stage5C or SMC
was executed. This completed experiment is not completion of the research goal.
