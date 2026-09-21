# Bounded Cost Study: Execution and Reproduction

Registration `3b6bb0a7` preceded the first real fit. The registered code/config
have not been changed after outcomes. Source bindings: 617. Runtime: native arm64
`.venv-pytorch`, CPU4/inter-op1/workers0. No new CREATE job or remote job-status
claim. Models, arrays and checkpoints remain local and are excluded from Git.

## Completed Execution

1. Preflight: PID85800, terminal exit0, 617 bindings verified.
2. Real pilot: PID85924, terminal exit0. First 100 updates of
   coupa/seed17/direct-native; finite loss and gradients, atomic checkpoint.
3. Resume: PID85985, terminal exit0. That same fit resumes; all 36 endpoints reach
   exactly 3,000 updates. Total 108,000 updates and 27,648,000 draws; unknown-label draws: 0.
4. First readout: PID86623, terminal exit0. Every new score and choice archive
   is fixed before loading evaluation outcomes. Analysis hash:
   `19bc91b35db63de0ce626f92f78094a05dccacf79d4426c119ea50a63d6858d5`.
5. Full replay: PID86673, terminal exit0. 36 checkpoints and 1,581,804 score rows;
   all 12 matched samplers agree. Original immutable analysis remains unchanged.
6. Separate verifier: terminal exit0. 108 policy constructions, 1,152 scene
   reductions and 3,885,786 repeated complete training rows check. It independently
   sorts the offline ranking and computes native errors and missing-step bounds.
   This is a separate implementation by the same agent, not independent research
   replication or an independent test cohort.

There are 22,978 parameters per cost head. Complete receipts preserve losses,
gradient norms, optimizer/sampler states and per-row draw counts. Reported fit
timers exclude data preparation and evidence validation; they should not be
used as end-to-end experiment wall time. The small head speed is not evidence
of having trained a new large forecaster.

## Commands

Run from the repository root, with existing verified private artifacts:

```sh
.venv-pytorch/bin/python scripts/run_m3w_bounded_cost.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_bounded_cost.py --resume
.venv-pytorch/bin/python scripts/run_m3w_bounded_cost.py --evaluate
.venv-pytorch/bin/python scripts/run_m3w_bounded_cost.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_bounded_cost.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_native_*.py tests/test_m3w_interaction_controls.py tests/test_m3w_joint_intervention.py tests/test_m3w_forecast_cost_bounds.py tests/test_m3w_bounded_cost_head.py
```

Completed fits are hash-verified and reused; `--resume` does not retrain completed
endpoints. Interrupted fitting resumes the last atomic checkpoint with optimizer
and random state. A changed identity is rejected. The file lock prevents two
writers; heartbeat records PID, fit and step. This is a local artifact-dependent
reproduction, not yet a download-and-run anonymous package.

191 scoped tests pass, including exact continuous/resumed training equivalence,
unknown-label exclusion, common initialization, zero-D behavior, matched-count
rules and analytic missing-outcome bounds. The legacy full suite is not rerun:
it contains integration training that can rewrite experiment reports. Previously
verified unchanged checks are reused; no claim that the entire repository suite
passed in this experiment.

## Interpretation and Recovery

The readout contains a positive primary accuracy contrast and a failed primary
protection criterion. The promising fraction objective is a fixed secondary
arm, not a retrospectively renamed primary outcome. All modes and seeds remain
in the report. No endpoint, threshold, sample population or risk tolerance was
chosen from closed test data. Four source sites remain design-exposed.

For future changes create a new experiment identity, preserve this one, and
state which observations motivated the amendment. No approved independent data
role is inferred from successful hashing. Do not delete caches to force a new
"first" readout. No deployment, metric/seconds claim, Stage5C or SMC follows.
