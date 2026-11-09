# Fitting-Only Forecasts for Learning the Selector

I rebuilt the forecast bank for learning the selector because the archived
alternative producer folds had trained on 6 and 8 sources in the later A
readout. Instead, I divided the nine nonempty fitting sources into three
held groups and fit PCT and EqMotion within their six-source complements.
The source groups, not individual rows, are excluded. The archived frozen
forecasts for A were not changed.

`fresh_run`: 18 native Torch fits, 10,800 optimizer updates, with both
families emitting one prediction for all 69,746 eligible fitting rows at
each of three seeds. Local wall time 1,444.51 s; maximum resident memory
1.41 GB. `cached_verified`: all checkpoint and prediction hashes, exact
training index and loss factors, zero held-source draws, and exact first
128 predictions per model (2,304 recomputed rows). This is not a claim that
every cached prediction was recomputed or that the shorter donor models
match the accuracy of the 4,000-update A models.

No A targets, independent confirmation, deployment, Stage5C or SMC. No
threshold or scientific role change. Data, weights and full forecasts stay
local; only their identities and light receipts are tracked.

```sh
.venv-pytorch/bin/python scripts/run_m3w_cv_joint_oof_bank.py
.venv-pytorch/bin/python scripts/verify_m3w_cv_joint_oof_bank.py
```

[Protocol](protocol.md) | [Freeze](freeze.json) | [Verification](verification.json)
