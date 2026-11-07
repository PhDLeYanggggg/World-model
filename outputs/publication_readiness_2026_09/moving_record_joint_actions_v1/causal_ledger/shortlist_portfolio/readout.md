# Cost Rankings Do Not Replace Family Diversity At Eight Slots

The new portfolio keeps the old top action, then cycles through
support, gain, risk-adjusted gain, and low-excess rankings, preferring
distinct agents. The original forecast bank, eight-slot budget,
direct interventions, 2% rules, and source roles are unchanged.
This is the exposed Moving A 8/12 annotation-step development task
in native image coordinates.

In the first two eligible queries of the 18 original zero-direct
views, the full bank has a useful choice in 15 views (verified prior
result), the family-diverse shortlist in 6, and the portfolio in 3.
The two lists share one hit: the portfolio adds two views and loses
five. Hence retrieval fails its descriptive comparison. The refit
risk winner hits 0/18; the event winner hits 1/18 without thresholding.
The latter is not a complete policy pass.

| Policy | Calibrated switches | Complete passes / 102 | Signed easy >2% | Undefined easy upper |
|---|---:|---:|---:|---:|
| Direct only | 6,952 | 2 | 8 | 35 |
| Family-diverse, risk winner | 6,955 | 2 | 8 | 35 |
| Portfolio, risk winner | 6,961 | 2 | 8 | 35 |
| Portfolio, event winner | 6,955 | 2 | 8 | 35 |

| Portfolio choice | Calibrated paired effect [95% interval] | Same 6,998 switches [95% interval] |
|---|---:|---:|
| Risk winner | -0.00461 [-0.01417, +0.00120] | -0.01471 [-0.04221, +0.00148] |
| Event winner | +0.00061 [0, +0.00182] | -0.01223 [-0.02959, -0.00100] |

Effects are percentage points of completion-lower utility versus
the family-diverse risk winner. Bootstrap intervals use 3,000 paired
source draws over 14 sources, averaging three seeds within source.
At matched queries and counts, complete passes are 5/102 for the
family-diverse control, 2/102 for portfolio risk, and 3/102 for
portfolio event. These are diagnostic, not a safety guarantee.
Both calibrated portfolio policies fail the registered improvement
criterion. No promotion or external outcome access.

The new rankings expose some different opportunities, but replacing
family diversity costs more opportunities than it gains. This
supports keeping family coverage in the current shortlist; it does
not establish that eight slots are sufficient, or that no other
retrieval or ranking rule can succeed.

## Reproduction

Registration `5cd2900e` precedes the 42 fresh cost-head fits. Each
cell uses seven fitting sources, six calibration sources, and one
held source. The two choice rules share a fitted head but have
separate donor-calibrated thresholds. The forecaster outputs are
verified parents, not new training. From the repository root with
the private forecast banks available:

```sh
.venv-pytorch/bin/python scripts/run_m3w_shortlist_portfolio.py fit
.venv-pytorch/bin/python scripts/run_m3w_shortlist_portfolio.py score
.venv-pytorch/bin/python scripts/run_m3w_shortlist_portfolio.py replay
```

Fitting resumes from completed source checkpoints. Scoring verifies
unchanged direct actions, identical matched query counts and prior
control recall. The score replay reproduced the analysis hash.
No rules changed. No independent confirmation, deployment, metric
or seconds claim, Stage5C execution, or SMC.
