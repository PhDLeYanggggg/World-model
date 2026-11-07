# Ranking Improves Shortlist Recall But Not The Opening Policy

Eighty-four real Torch CPU training jobs complete 140 updates each:
two objectives in each of 42 cells from 14 held sources and three
forecaster seeds. Every cell uses seven fitting sources, six disjoint
calibration sources and one held source. The small shared network,
training draws, initialization and trajectories match between arms.
The ranker adds a within-query logistic pair loss to the pointwise
objective. It follows the established pairwise scoring idea in
[RankNet (Burges et al., 2005)](https://doi.org/10.1145/1102351.1102363),
not a new generic ranking loss. The difference tested here is ranking
positive, supported trajectory substitutions in a frozen candidate bank.

## Shortlist Recall

First two eligible queries of the 18 original zero-direct views.
The full bank contains a strictly positive, supported action in 15.
Counts are views with at least one such action; a shortlist hit is
not a causal choice of that action.

| Score | Flat top 1 | Flat top 8 | Distinct rows first, 8 slots | Distinct families first, 8 slots |
| --- | ---: | ---: | ---: | ---: |
| Previous joint score | 0 | 2 | 1 | 6 |
| Pointwise useful-action model | 1 | 4 | 4 | 4 |
| Within-query pair loss | 1 | 5 | 6 | 5 |

The diversity experiment is a fixed-budget, posthoc diagnostic, with
no policy or threshold retuned. It recovers the flat controls exactly.
Across all eligible queries, flat top-eight recall is 4, 8 and 9
respectively. The pair loss increases some shortlist counts but gives
no top-one advantage over pointwise in the first two queries. Simple
family diversity on the previous score matches the best eight-slot
first-two-query recall of the learned rankers.

## Observed Policy Results

| Policy | Switches | Full original passes / 102 | Easy net >2% | Zero-direct opened / passed |
| --- | ---: | ---: | ---: | ---: |
| Direct | 6,952 | 2 | 8 | 0 / 0 |
| Calibrated pointwise | 6,959 | 2 | 8 | 1 / 0 |
| Calibrated pairwise | 6,953 | 2 | 8 | 0 / 0 |
| Diagnostic previous joint | 6,998 | 5 | 11 | 18 / 3 |
| Diagnostic pointwise | 6,998 | 2 | 11 | 18 / 0 |
| Diagnostic pairwise | 6,998 | 2 | 12 | 18 / 0 |

At identical diagnostic queries and counts, the pairwise contrast
against the previous joint score is **+0.01276 pp** of completion-lower
ADE utility (95% source interval **[-0.02227, +0.06424]**); against
pointwise it is **-0.00134 pp** (95% interval **[-0.00361, +0.00026]**).
Neither supports a deployment improvement. The policy still uses
the original 2% observed rule. The next test should separate budgeted
shortlist construction from actual risk-aware choice, using the
simple family-diverse shortlist as a control.

## Reproduction

Native arm64 Torch 2.12.0, four threads, no loader workers. Fit and
calibration take 51.6 s. All 84 final optimizer checkpoints were
checked for the registered seeds, steps and loss weight. The update
resume test reproduces continuous training exactly; the full scoring
replay is byte-identical. 91 scoped tests pass. The task remains 8/12
annotation steps in native image coordinates. No independent outcome
access, deployment, Stage5C or SMC.

[Analysis](analysis.json) | [Diversity audit](diverse_shortlist.json)
