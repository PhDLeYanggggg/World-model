# Waiting Does Not Repair A Missing Top-Ranked Action

Fresh stop and continuation fits on the same source-separated Moving A
development task: eight observed and twelve future annotation steps in
native image coordinates, 34 recordings, 14 sources and 102 views from
three forecaster seeds. Forty-two cells use 7 fitting, 6 calibration and
1 held source. The two policies share joint candidate ranking and the
frozen trajectories. The direct decisions and original 2% observed
risk/support rules are unchanged. The fitted decisions do not see the
next query or any test outcome.

| Opening policy | Switches | Original passes / 102 | Signed easy >2% | Zero-direct views opened / passed |
| --- | ---: | ---: | ---: | ---: |
| Direct only | 6,952 | 2 | 8 | 0 / 0 |
| Calibrated first stop | 6,952 | 2 | 8 | 0 / 0 |
| Calibrated stop-wait | 6,952 | 2 | 8 | 0 / 0 |
| Diagnostic first stop | 6,998 | 5 | 11 | 18 / 3 |
| Diagnostic stop-wait | 6,984 | 2 | 11 | 18 / 0 |

Each calibrated arm abstains in 41 of 42 threshold cells. The remaining
threshold also triggers no held-source action. The calibrated paired
utility contrast is exactly zero because the actions are unchanged,
not evidence of equivalence. The unthresholded stop-wait moves the
opening to the later query in 32 views and adds fewer interventions.
It loses **0.01697 pp** of paired completion-lower utility versus
unthresholded first stop (95% source interval **[-0.04101, +0.00395]**).
This is not a matched-count utility comparison. The three diagnostic
first-stop passes in the zero-direct slice have no strictly positive
supported single action; a risk pass need not imply utility gain.

## Ranking Availability

We separately scan all 18 zero-direct views with the existing outcome
labels, without changing any rank or policy. A hit requires a strictly
positive single-action ADE gain and the unchanged standalone support
and risk rules. Each column counts views with at least one hit across
the indicated queries, not actions or independent recordings.

| Eligible queries searched | Entire bank | Joint top 1 | Joint top 3 | Joint top 8 |
| --- | ---: | ---: | ---: | ---: |
| First | 9 | 0 | 0 | 0 |
| First two | 15 | 0 | 0 | 2 |
| All | 15 | 0 | 2 | 4 |

The marginal and original-utility scores also have zero top-one
recall, even when allowing all queries. Therefore, on this exposed
slice a rule restricted to one top-ranked opening cannot recover a
positive supported action by timing alone, even with perfect timing.
The entire bank still contains such a choice in 15/18 views.
This does not establish what a learned reranker or multiple openings
would achieve. The next comparison should target candidate recall
and rank before training a more complex timing rule.

Regression of continuation value is an established stopping idea
([Longstaff and Schwartz, 2001](https://doi.org/10.1093/RFS/14.1.113)).
Here the future training target is the realized return of a next-query
stop rule on fitting recordings, not the maximum realized future reward.
The exact recording-level trajectory scorer still determines risk;
we neither reproduce the finance method nor claim its guarantees.

Native arm64 fit: 51.0 s. Source-separated fitting query counts range
from 10 to 170 (median 39.5). All direct controls and the complete
scoring replay match exactly. 88 scoped tests pass. No independent
outcome access, deployment, Stage5C or SMC.

[Scoring](analysis.json) | [Rank recall](rank_recall.json)
