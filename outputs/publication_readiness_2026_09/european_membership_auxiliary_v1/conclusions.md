# Auxiliary Membership: Completed, Strong Cost Gate Failed

## Material Passport

Fresh_run:288 real Torch fits,576,000 updates,36 result groups and144 held
folds. Cached_verified: source features/producers and original cost models.
Registration cadc04df preceded fitting; freeze8daab80d was pushed before
held readout. Pure fitting totaled541.7740seconds, excluding loading,
inference and verification. Unknown-label training draws:0. No new policy,
independent selection/calibration/confirmation or deployment.

## Main Result

The matched cost-only control reproduces original held cost predictions
exactly(maximum absolute difference0). Adding membership auxiliary loss
does not pass the registered expected-cost gate:

| Full positive-disagreement comparison | Positive / negative / overlapping95% intervals | Assignment point range |
|---|---|---|
| Auxiliary vs matched control |1 /1 /4 | -8.1329% to +1.0446% |
| Auxiliary vs original |1 /1 /4 | -8.1329% to +1.0446% |
| Auxiliary vs failed conditional model |3 /0 /3 | -45.4408% to +18.7896% |

These are easy-harm MSE changes, not trajectory ADE/FDE gains. A win over
the failed conditional model cannot rescue the strong-comparator failure.
Full top10 harm-capture contrasts vs original have2 positive,1 negative and
3 overlapping intervals: the tail guard fails. All-harm MSE intervals all
overlap zero(points -23.38% to +2.19%); that is not proof of noninferiority.
Motion-only easy-harm comparisons retain2 positive,2 negative and2 overlapping
intervals(points -10.58% to +5.18%). No favorable-source selection.

## What Was Learned

The auxiliary task is genuinely learned: across72 dependent full held views,
median membership AUROC is0.86123, Brier0.10529 and log-loss0.33882. The
untrained control branch hasAUROC0.5, Brier0.16679 and log-loss0.52195.
These descriptive medians do not constitute a new classification gate or
policy result. The strong membership signal still fails to improve cost
magnitude consistently. In full views, cost fitting improves in21/72 and
held MSE in40/72;12 of the fitting improvements do not transfer. This is
not simply a case where a uniformly better fit overfits held scenes.

The experiment rejects this fixed auxiliary-loss repair, not all forms of
multi-task learning or all future cost predictors. Next diagnostic: measure
cost/BCE gradient alignment and magnitude on frozen fitting batches before
choosing a single decoupling or weighting repair. Do not start a held-score
weight sweep or change the policy threshold.

Three seeds averaged per locality,3,000 resamples of four localities per
source assignment. Roles/windows are dependent, historically exposed,
exploratory and not multiplicity-adjusted. No independent-confirmation or
submission-ready claim. Pixel/annotation-step only, detector-derived labels;
no metric/seconds, human gold, physical safety, true3D or foundation claim.
Stage5C/SMC remain off. The research goal is ongoing.
