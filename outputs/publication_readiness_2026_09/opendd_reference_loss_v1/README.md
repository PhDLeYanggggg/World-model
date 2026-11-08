# Predicting Whether an Early Edit Erases the Reference

[Earlier support](../opendd_early_reference_v1/README.md) produced
86 changed proposals and erased 19 defined easy references.
This study holds the gain edit, early proposal, time and path fixed.
The guard sees the proposed path, current motion and summaries of
already observed queries. It predicts loss directly or as the
probability of an old unique reference multiplied by the conditional
probability that the candidate contributes none. Future outcomes
appear only in the training target and evaluation.

We keep proposals at or below the 75th percentile of the risk
score on in-sample donor proposals, with no held-location input
to the cutoff. This is a training-side budget heuristic, not an
independent safety calibration. Rejection leaves the original
support trigger intact. The retrospective easy-score controls
match accepted counts separately for additions and replacements
in each site and seed. They use the same frozen menu but can
choose other recordings; they do not match every radius.

| Policy | Accepted | Edits | Joint pass | Risk pass | Undefined | Old ref lost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 0 | 943 | 138 | 172 | 295 | 0 |
| All early proposals | 86 | 984 | 131 | 164 | 305 | 19 |
| Direct guard | 62 | 972 | 134 | 169 | 302 | 13 |
| Easy, matched to direct | 62 | 972 | 134 | 167 | 301 | 14 |
| Factored guard | 64 | 972 | 135 | 170 | 302 | 13 |
| Easy, matched to factored | 64 | 972 | 133 | 166 | 302 | 15 |
| Exact-count hindsight | 62 | 972 | 140 | 173 | 300 | 11 |
| No-loss hindsight, up to count | 35 | 962 | 142 | 176 | 293 | 0 |

The 606 views are exposed recording-seed views from seven locations,
not 606 independent recordings. The direct guard retains 29
additions and 33 replacements. Its paired conservative utility is
-0.00186 pp [-0.00425, -0.00014] versus the frozen early policy and
-0.00023 pp [-0.00092, +0.00030] versus matched easy ranking.
Against the original it is +0.00015 pp [-0.00303, +0.00269].
The intervals resample locations 3,000 times after averaging seeds.
The guard has not shown a utility-and-risk improvement.

## How Much Can Thinning Recover?

For each view, the audit scores the exact original and early
trajectories. Their effects on joint pass add across recordings,
so sorting the pass changes within each site/seed/add-replace stratum
maximizes passes at the exact learned count. The original scorer
rechecks the resulting plans. Without reference loss, seven slots
cannot be filled at those quotas. Allowing underuse achieves 142
joint passes, just four more than the original. Hindsight scores
are menu limits, not causal policy outcomes or deployment evidence.

The fixed menu provides little room for further threshold search.
The next experiment should broaden the *generation* of support
proposals at matched motion and intervention budgets, then test
whether the added feasible actions are recoverable with observed
history. Keep a candidate for independent calibration only if
it beats its simple matched controls on utility and risk.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_reference_loss.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_reference_loss.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_opendd_reference_loss.py
```

`fresh_run`: 21 fits, 10,500 updates, native arm64 Torch, four
threads, zero loader workers; PID 79090, 15.8 seconds. The witness
audit performs no training. `cached_verified`: frozen proposal
generator and zero-update replay. `not_run`: new CREATE training
and independent confirmation. The per-recording 2% rule and
undefined-risk handling are unchanged; the 30 DroneCrowd confirmation
groups remain sealed. No deployment change, submission, Stage5C or SMC.
