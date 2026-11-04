# Why Source-Separated Neural Critics Did Not Repair Admission

Completed real training, full nested-C scoring and exact replay; no new H
readout or deployment. Original rules were retained at the user's explicit
request. This is a negative development experiment, not independent proof
that every possible source-separated critic must fail.

## What Changed

There are 108 new cost heads and 216,000 saved primary updates. Each model's
labels, feature normalization, target RMS and bias use only its own inner
source part. Proposer and disjoint risk estimator exclude the validation
part and each other's fitting sources. Same-data/different-seed critics are
retained as controls. Every query's matched control has the same count of
each amplitude as its veto. Thus the comparison is not merely switching less
or using a different amplitude mix. Displacement magnitudes are not matched.

This construction changes fitting-source coverage and own-source scaling;
it does not identify a pure causal effect of statistical independence.
All models still share a forecast family, architecture and exposed dataset.

## Measured Failure Taxonomy

|Failure|Evidence|What it establishes|
|---|---|---|
|Primary lacks intervention support|Disjoint: 18/276 required source views empty; same-data: 6/276|The original complete nonempty primary and its CI are unavailable, not zero or passed|
|Veto removes too much useful benefit|Disjoint full-strategy known contrast -0.56277 pp [-1.09473, -0.19341]; direct lower -1.27394 pp [-2.08928, -0.58580]|The negative secondary cannot rescue the unavailable primary|
|Harm reduction is smaller than benefit lost|Known harm reduced 0.21292 pp [0.09323, 0.34611]; benefit lost 0.77569 pp [0.35576, 1.32480]|Both use the same full-strategy denominator; do not combine either with nonempty-primary statistics|
|Selected cost estimates remain optimistic|On identical known selected rows, actual/predicted H mass 122.85; EH mass 553.27|Separating fitting sources did not calibrate the selected-risk estimates|
|Full-policy risk still fails|44/696 supported; 443 known all-risk and 448 known easy-risk violations|This is not just missing denominators; many estimable risks actually exceed 2%|
|Support remains independently incomplete|173 selected easy-risk views undefined; 91 record views have no intervention; 30 have zero full known easy reference|These counts overlap and must not be added; zero full reference cannot be manufactured by training|

The worst defined signed recording-level easy-net degradation is 477.76%.
This is not a dataset-wide mean. Empty nonempty-primary source views are:
component049 (12 disjoint, six same-data), component067 (three disjoint),
component076 (one disjoint) and component082 (two disjoint). They remain in
the required population. No intervention is forced merely to fill a control.

The source-disjoint arm has positive known gain against its physical floor
(3.32980%), but its count/amplitude-matched control gains 3.89258%. A positive
comparison with the weaker floor is not evidence that this veto adds value.
Likewise, a falling fitting loss is not validation accuracy or a risk bound.

## What Is Not Proven

The mass ratios pool repeated selected rows; they are not source-level
confidence bounds or probability calibration scores. They do not by
themselves isolate missing features, camera motion, model capacity, smaller
fitting-source coverage, or optimization as the unique causal explanation.
Do not directly compare these source-excluded ratios with old supervised-C
ratios on different choices and call the difference a measured domain shift.
No claim is made that source splitting is theoretically useless.

## Next Model Hypothesis, Not Yet Trained

The measured bottleneck is selected harm estimation, not another confidence
threshold. The next controlled model candidate should explicitly separate
positive-harm event learning from conditional harm magnitude, including an
easy-harm branch, and compare it with this frozen point-moment bank. Test
whether it improves discrimination and mass reliability rather than simply
inflating risk everywhere and rejecting all actions. Reference overprediction
must be measured on the same selected rows too.

Use a real fit-only numerical/resume pilot before a larger run. Keep the
original data roles, risk limits and all required views; define the loss and
comparison before new scores. Do not reuse H to choose the model, force
unsafe switches to populate an estimator, add identity certification, or
relabel exposed sources as confirmation. This is a next hypothesis, not
completed training, a guaranteed repair or permission for Stage5C/SMC.
