# Cost-Sensitive Fitting: More Benefit, Still Unsafe

29 October 2026. Fresh_run: 18 real neural policy fits, 36,000 updates, all
72 fitting readouts and an exact replay. Cached_verified: source packets,
forecasts, causal features, moment predictions and frozen comparator models.
Not_run: new inner-held scores, independent selection/calibration/confirmation,
new trajectory forecasting, or deployment.

## What Changed

I replaced the sigmoid-action straight-through primal loss with a cost-weighted
logistic loss. The causal inputs, 385-to-32-to-1 policy, initialization, update
budget, record weighting, dual updates, 2% safety budget and logit threshold
were fixed. All final checkpoints were committed in `01ba5415` before readout.
There was no selection of a best seed, checkpoint or decision threshold.

## What Worked

|Fitting diagnostic|Frozen exact-gradient policy|Cost-sensitive policy|
|---|---:|---:|
|Eligible known positive benefit retained|25.998%|81.552%|
|Known benefit / full known reference cost|5.111704 pp|16.034646 pp|
|Known harm / full known reference cost|0.193142 pp|0.547302 pp|
|Aggregate source/context/seed views within budgets|53/72|54/72|
|Recording/context/seed views within budgets|178/444|84/444|

Cost masses are normalized by full known reference cost in each view and then
macro-averaged over the 72 views. Benefit retention is the ratio of these means
to the identically normalized eligible-positive-benefit upper bound. It is not
a count of correctly classified rows or a pooled trajectory-error percentage.

At matched intervention counts, the cost-sensitive policy improves TRAIN
utility over the exact-gradient control by a completion-bound interval of
**[+1.824042, +1.824777] pp** of full known reference cost. The unmatched interval
is [+10.567627, +10.569935] pp. The matched result shows that simply making more
interventions is not the entire explanation of the gain against this control.
These are bounds over unknown outcome completions, not confidence intervals,
FDE improvements, held-data estimates, or safety certificates.

Against the original frozen policy, matched bounds are [-1.263524, +0.981568] pp;
superiority is not established. Three matched views against exact-gradient are
empty. They stay in utility accounting with risk not_estimable, never safe zero.

## What Failed

The full required recording roster does not pass. Known all-positive-harm
violations increase from 216 to 346 of 444 repeated record views. Of the new
346 violations, 343 are in the two multi-recording sources locality008 and
locality074. Aggregate source performance conceals this concentration.
There are 112 formerly passing record views that now fail, and 18 newly passing
ones. Empty easy-risk counts fall from 55 to 14, so missing support is not the
main explanation for this worsening. Net easy degradation stays within 2% in
all 444 views, but that does not waive the separate positive-harm constraints.

The known locality082/51 fixed-action obstruction also remains: neither of its
forecast contexts has an eligible known easy-reference action within 2%.
That recording is not removed. No selector alone can make its current fixed
candidate set satisfy the supported-selection requirement.

## Mechanism and Next Step

Wrong saturated decisions fall from 443,891 to 25,925 repeated row occurrences
at the models' own final TRAIN dual states. This is consistent with reducing
the old surrogate's saturation problem; it is not a unique causal attribution.
The final-update diagnostic still has 344 positive all-risk record constraints
and zero all-risk duals at the cap. Merely raising the dual cap would not address
an active clipping problem in that channel. Finite-time constraint optimization,
record weighting and shared-model compromise remain hypotheses to distinguish.

The next bounded experiment should retain the improved benefit-learning loss
and test recording-constraint optimization, not rescue it with held thresholds.
In parallel, the unsupported recording needs a better legal forecast candidate
or additional valid support, not relabeling an empty comparison as safe.
These follow-up training changes are not_run in this experiment. All subsequent
readouts must keep every recording, unchanged budgets and explicit data roles.

## Evidence Boundary

There are 74 unique source-qualified recordings, appearing in 148 forecast
contexts and then three head seeds. The 444 views and repeated windows are not
independent samples. No new bootstrap is reported for this fitting-only repair.
Training took 604.904s with peak RSS 3,015,131,136 bytes. Final model artifacts
and completion heartbeat were verified; the original training terminal exit
code was not retained. The readout and replay both completed with exit 0.
The report independently reconciles 1,836 policy summaries; 57 targeted tests
pass. The full historical test suite was not rerun.

Deployment remains unchanged. This is not submission-ready evidence, true 3D,
a foundation world model, a metric/seconds claim, Stage5C execution or SMC.
See [complete readout](readout.md), [constraint diagnosis](constraint_diagnosis.md),
[failure analysis](failure_analysis.md), [model card](model_card.md) and
[Chinese reproduction guide](operation_zh.md).
