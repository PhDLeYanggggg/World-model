# Moving: what changes when the same budget can move across queries?

I kept the old half-peak count and displacement ceilings and let
the edits move across queries. This tests whether the earlier
single-query failure came from where I spent the budget or how
much improvement that budget could buy.

**Fresh run:** one arbitrary-path bound and a four-arm allocation
audit on 72 recording-seed views (24 recordings, nine development
sources, three seeds). **Cached verified:** the exact v155 held
paths, folds and old anchor. No new training. The first local
solve finished in 131 seconds; replay rebuilt the bounds and
checked the saved witnesses. A smaller integer search around
the fractional solutions resolved all three previously open views.
The follow-up can verify a witness but cannot exclude a plan.

| Available actions | Budget stays inside each query | Same ceilings pooled by recording |
| --- | ---: | ---: |
| Old half-peak anchor | 2/72 | 2/72 |
| Base paths | 4/72 | 5/72 |
| Add v155 paths | 4/72 | 5/72 |

The counts are exact for these menus and budgets: each pass is
replayed with both conservative gains positive, positive easy
reference, and the original all/easy/signed 2% rule. For the
query-fixed arm, 67 views are excluded by the arbitrary-path
bound and one more by the finite-menu upper bound. For pooling,
67 are excluded by the arbitrary-path bound. No solver timeout
is counted as a negative result.

In the one view that needs pooling, the base menu achieves
+1.046 pp vs CV and +0.031 pp vs OLS4, with 1.421% easy positive
harm and -1.313% signed easy degradation. Adding v155 paths
allows a different witness (+0.059 pp vs OLS4, 1.974% easy
positive harm) at the same ceilings, but no additional pass.
These are outcome-aware witnesses, not predictions made online.
The 72 views are not 72 independent recordings.

**Budget and missing labels.** Even without risk, edit counts,
or a finite path menu, OLS4 cannot be beaten in 67/72 views at
either timing budget. The median OLS4 upper gain improves from
-29.82 to -27.04 CV-normalized pp under pooling. Over the 62
nonzero-budget views, the median minimum peak mass to tie
OLS4 rises from 1.02 times the current budget on available
positions alone to 18.71 times when missing steps are debited.
The corresponding median components of the CV-to-OLS4 gap
are 5.61 and 34.23 pp; they are computed separately, not added
as medians. Ten views have zero original budget and are not
used for budget ratios. This is an unrestricted lower limit
on a necessary budget, not a feasible operating point and not
an estimate of error at unlabeled positions.

I would not spend another full run on ranking the same small
CV edits. For the next comparison, I will first check whether
a causal forecast family closer to the strong reference can
increase the usable capacity under a matched budget, and keep
the better-supported label sources separate from this exposed
Moving roster. Deployment is unchanged. No Moving A or
DroneCrowd confirmation outcomes were opened, and Stage5C
and SMC remain off. The trajectory protocol uses image pixels
and annotation steps, not metric distances or seconds.

[Protocol](protocol.md) | [Initial solve](readout.json) |
[Repair and exact checks](repair.json) | [Break-even diagnostic](break_even.json) |
[Manuscript v156](../evidence_manuscript_v156/README.md)
