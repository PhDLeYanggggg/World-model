# Conditional Risk Choice Does Not Convert Shortlist Recall Into Action

Same exposed Moving A development task: eight observed and twelve
predicted annotation steps, image coordinates, 14 sources, 34
recordings and 102 views across three forecaster seeds. The eight-slot
family-diverse list reproduces the previous shortlist control. Forty-two
fits use 7 training, 6 calibration and 1 held source. A single fitted
cost head supplies both learned selectors. They differ only in whether
positive predicted risk costs are subtracted from predicted lower gain.

| Policy | Switches | Complete original passes / 102 | Easy net >2% | Zero-direct opened / passed |
| --- | ---: | ---: | ---: | ---: |
| Direct | 6,952 | 2 | 8 | 0 / 0 |
| Calibrated gain | 6,959 | 2 | 9 | 2 / 0 |
| Calibrated risk | 6,955 | 2 | 8 | 0 / 0 |
| Diagnostic previous joint | 6,998 | 5 | 11 | 18 / 3 |
| Diagnostic gain | 6,998 | 5 | 11 | 18 / 3 |
| Diagnostic risk | 6,998 | 5 | 11 | 18 / 3 |

The three diagnostic policies match switch counts query by query.
Risk minus gain changes paired completion-lower utility by **+0.00156 pp**
(95% source interval **[-0.00688, +0.01348]**). The risk term does not
establish an improvement at fixed coverage or after calibration.
The calibrated risk arm abstains in 39 of 42 threshold cells.

## Available Versus Chosen

In the first two eligible queries of the 18 original zero-direct views,
the full bank has a positive supported action in 15, and the family
shortlist retains one in 6. Both learned selectors choose a useful
winner in 0/18 when tested on that shortlist. In a separate posthoc
readout of the same frozen head, ranking directly by its predicted
useful-event score reaches 1/18. That choice changes 34 diagnostic
views at the same total and query counts; full passes drop from 5 to 3,
while signed easy violations fall from 11 to 9. Its small utility
difference (+0.00439 pp) is exploratory and does not support promotion.

These results distinguish shortlist coverage from choice quality.
The predicted costs do not reliably separate the few useful openings
from other actions. Further weight tuning is a weaker next test than
adding past-observable evidence of recent forecast errors and evaluating
it against the same frozen list. No future-label inputs or independent
outcomes are used to rank held queries. The original 2% rule is unchanged.

## Verification

Native arm64 fits complete in 60.9 s; all 42 role partitions are checked.
Direct controls match exactly and the full score replay is byte-identical.
93 scoped tests pass. No deployment change, Stage5C or SMC.

[Analysis](analysis.json) | [Event-score diagnostic](event_choice_diagnostic.json) |
[English appendix](../../../evidence_manuscript_v10/shortlist_appendix.md)
