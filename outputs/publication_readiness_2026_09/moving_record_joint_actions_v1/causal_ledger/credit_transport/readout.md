# Predicted Ledger Credit Does Not Carry Over To Realized Costs

The chronological ledger improved matched utility, but it assumes that
estimated savings can fund later switches. I audited this assumption on
the frozen actions and fitted a separate discount for all and easy credit.
For each held source, donor predictions come from fits excluding both that
source and the donor. The four arms have the same 6,184 interventions and
exactly the same count in each query. No original thresholds or costs
within a query change. Three existing forecast seed banks, 42 new discount
fits, no new trajectory model.

| Accounting on queries predicted to earn credit | All | Easy |
|---|---:|---:|
| Predicted earning queries | 2,601 | 1,810 |
| Queries actually incurring positive completion excess | 1,310 | 600 |
| Predicted savings mass | +1,497.17 | +131.63 |
| Realized signed savings on those queries | -6,149.42 | -2,609.64 |

Here positive savings means negative excess under the original budget.
These masses are pooled repeated-query diagnostics in image-error units,
not risks or independent sample counts. The realized term includes the
original completion radius where labels are wholly missing. On 703 of the
1,631 queries where the full ledger spends prior credit, the same budget
dimension has positive realized excess. That is not an assertion about a
prefix passing the recording-level observed test.

| Carry policy | Completion lower vs CV (pp) | Original passes / 102 | Undefined easy |
|---|---:|---:|---:|
| None | +0.08275 | 2 | 36 |
| Full | +0.10154 | 3 | 34 |
| Fixed half | +0.09793 | 2 | 35 |
| Nested donor discount | +0.08275 | 2 | 36 |

All 42 fits retain zero credit for both budgets. The donor policy exactly
replays the no-carry actions. Its direct paired lower versus full carry is
-0.02135 pp [-0.03808, -0.00538]. Half carry versus full is -0.00420 pp
[-0.00988, +0.00051], with no improvement in observed passes. Intervals
use 3,000 paired draws of the 14 sources after averaging three seeds.
This tests a specific conservative empirical discount, not all possible
credit models. It does not validate a useful safety correction.

**Decision.** The earlier chronology-correct utility signal remains, but
estimated savings cannot be treated as certified credit. The next research
question is whether selected-cost estimates can transfer under separately
assigned calibration and confirmation roles, not which credit fraction
looks best on A. No independent outcome or deployment change here.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_ledger_credit_transport.py fit
.venv-pytorch/bin/python scripts/run_m3w_ledger_credit_transport.py score
.venv-pytorch/bin/python scripts/run_m3w_ledger_credit_transport.py replay
.venv-pytorch/bin/python scripts/build_m3w_ledger_reproduction.py --verify
```

Protocol `5b193a93`; all native arm64 local processes exit 0. Calibration,
score and exact score replay take 37.19/11.52/11.45 s. Atomic seed checkpoints,
no workers, peak RSS 2.75 GB. The identity-stripped 232,609-byte aggregate
bundle regenerates 21 contrasts, 13 policy summaries and six credit
quantities in an isolated standard-library run. It is not full training
reproduction or an anonymity guarantee. It stays local with no raw data,
predictions, weights or model coefficients. No Stage5C or SMC.

[Protocol](protocol.md) | [Metrics](analysis.json) | [Portable verification](portable_verification.json)
