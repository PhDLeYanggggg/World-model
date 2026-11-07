# Missing-Label Debits Explain Some, Not All, Capacity Failures

This registered 2x2 diagnostic reuses the original native bank and
the exact 9,675-switch budget. The original evaluation charges
forecast disagreement on wholly unlabeled rows in two places:
positive all/easy harm and paired conservative utility. We set
each debit to zero separately, then both together, while preserving
support, labeled outcomes, the observed denominators and the
original signed-easy check. No diagnostic pass changes the
original evaluation or the independent protocol.

| Development arm | Verified both references | Negative maximin bound | Risk/count infeasible | Unresolved |
| --- | ---: | ---: | ---: | ---: |
| Original | 51 | 45 | 6 | 0 |
| Utility debit omitted | 63 | 33 | 6 | 0 |
| Positive-harm debit omitted | 57 | 42 | 3 | 0 |
| Both debits omitted | 66 | 33 | 3 | 0 |

For 6 of the original 45 negative views, either omission alone
suffices; for 6 others only the utility omission alone suffices.
Three originally risk/count-infeasible views need both omissions.
Of the original 51 failures, 36 still do not pass even with both
debits zero: 33 have a negative objective bound and 3 remain
risk/count infeasible. This is a statement about a diagnostic
with neutral accounting on unlabeled rows, not an upper bound
on utility after actual annotation.

The 15 newly verified diagnostic views repeat five recordings
from three sources. Source-balanced positive-rate changes against
the original are +11.43 pp [0.00, 28.57] for utility only,
+8.57 pp [0.00, 24.29] for risk only, and +12.86 pp [0.00, 30.00]
for both. Each paired 95% percentile interval resamples the 14
source groups 3,000 times after averaging the three seeds. The
intervals include zero and are not independent-validation evidence.

Reducing missing-label uncertainty is relevant to the two-reference
comparison. But neutralizing these debits is not a
substitute for better forecasting on the available data: it does
not make most strict failures positive. A diagnostic with the
original unknown-label debit disabled is never eligible for
deployment.

[Protocol](protocol.md) | [Registration](registration.json) |
[Analysis](analysis.json) | [Score receipt](score_execution.json) |
[Replay receipt](replay_execution.json).

```sh
.venv-pytorch/bin/python scripts/audit_m3w_missing_debit_capacity.py score
.venv-pytorch/bin/python scripts/audit_m3w_missing_debit_capacity.py replay
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_missing_debit_capacity.py \
  tests/test_m3w_dual_reference_capacity.py tests/test_m3w_ols_blend_capacity.py \
  tests/test_m3w_recording_risk_attribution.py
```

The initial run solves the alternatives and verifies each retained
witness with the applicable original or diagnostic evaluator.
Replay re-scores the positive witnesses and reproduces the aggregate;
it does not rerun the proof solves. Twelve targeted tests passed.
Data are exposed development, measured in image pixels at eight
observed and twelve predicted annotation steps. No new training,
external access, deployment, Stage5C execution or SMC.
