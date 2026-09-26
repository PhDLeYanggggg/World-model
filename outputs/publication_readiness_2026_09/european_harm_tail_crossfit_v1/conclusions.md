# Harm Tail Crossfit: What the Experiment Establishes

## Result

The risk features contain useful costly-tail information, but neither reliable
within-disagreement event discrimination nor transported harm magnitude is
established. The diagnosis does not support the simple claim that ranking is
already solved and only a global calibration multiplier is missing.

This is a completed training/readout diagnostic, not a new deployment result.
Registration `c666dac7` preceded the 144 cold-start Torch fits. Prediction
freeze `97df3dcb` was pushed before held-locality readout. All 288,000 updates
completed, with zero unknown-label draws. No forecaster, threshold, primary
endpoint, risk tolerance or reserved-data role changed.

## Useful Signal and Its Limit

For the full forecast pair, the learned harm moment captures more harm in its
highest-scoring 10% than the causal disagreement envelope. All six source-role
point contrasts are positive, ranging from +8.11 to +75.77 percentage points;
four locality-bootstrap intervals exclude zero and two overlap it. The
corresponding motion-only result has two positive intervals and four overlaps,
with points from -5.60 to +29.51 pp. These are tail-capture differences, not
trajectory improvements or deployment gains.

Within the positive-disagreement subset, full moment-versus-envelope points
are +4.48 to +59.38 pp: three positive intervals, three overlaps. Motion-only
has five positive intervals and one overlap. This is useful signal against
that control, not proof that every harmful event is ranked correctly.

| Held-locality diagnostic, median over 72 dependent views | Full | Motion only |
|---|---:|---:|
| Harm-event prevalence, all rows | 2.228% | 0.403% |
| Event AUROC, all rows | 0.79177 | 0.87301 |
| Event AUROC, positive disagreement only | 0.48578 | 0.55792 |
| AP / prevalence, positive disagreement only | 1.08796 | 1.47664 |
| Top10 harm captured by moment, all rows | 44.608% | 43.241% |
| Top10 harm captured by envelope, all rows | 9.375% | 25.797% |
| Predicted / actual harm mass | 0.66266 | 1.07677 |
| Oracle top1% share of total harm | 90.773% | 100.000% |
| Weak event-support views | 0/72 | 30/72 |

These medians describe different populations; they are not pooled estimates
over millions of independent observations. Harm occurrence ranking and harm
severity ranking are also different targets. A near-chance event AUROC can
coexist with some useful concentration of costly cases.

## Why the Aggregate AUROC Is Not Enough

Zero disagreement implies zero possible excess error for this forecast pair,
and the head outputs zero there by construction. The all-row population
therefore contains easy-to-separate structural negatives. Removing them
reveals much weaker event discrimination exactly where an intervention can
change the prediction. The full positive-disagreement median AUROC is 0.486;
this is a descriptive median, not a significance claim of worse-than-chance
performance. It rules out treating the all-row 0.792 as sufficient evidence
of a useful switch-risk classifier.

## Magnitude and Transport Remain Unresolved

Full held-locality harm is underestimated in 50/72 dependent views, with
median coverage 0.66266 and range 0.00207 to 4.26728. Thus a single scaling
factor cannot be justified from the median alone. The largest 1% of examples
carry a median 90.8% of harm; magnitude errors can dominate a decision budget.

Original full-B models are also reported with fresh B bins and hash-verified
C scores/labels. Their full positive-disagreement event AUROC falls from
median 0.51794 on fitting B to 0.47512 on C; top10 harm capture falls from
34.866% to 22.655%. B and C are descriptive source-development populations,
not a new paired independent generalization trial. Their whole-B easy cut is
not target-matched to the three-site cuts used in the new inner folds.

## Decision

No policy is promoted. No independent calibration or confirmation is opened.
The next controlled intervention should address risk discrimination on the
causal intervention-support population and its tail magnitude jointly, with
explicit controls for event prevalence, intervention count and prior failed
hurdle/ranking repairs. A global rescale, threshold sweep or another generic
oversampling run is not justified by these results. See failure analysis and
project gap for the specific remaining tests.

Image pixels, annotation steps and detector-derived labels only. No metric,
seconds, human-gold, physical-safety, true3D, foundation or submission-ready
claim. Stage5C and SMC remain off.
