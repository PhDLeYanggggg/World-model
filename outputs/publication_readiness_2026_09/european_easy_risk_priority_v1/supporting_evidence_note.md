# Supporting Optimization Evidence, Not a Main Efficacy Claim

This addendum is separate from the four-site SDD manuscript. Its population is
twelve already-opened European development localities, not an expansion of the
independent test set. Observation8/prediction12 with raw-frame stride12 and
image-local detector silver are retained.

In a fixed-forecaster experiment, explicit easy-occurrence and conditional-cost
supervision reduced occurrence Brier from0.2621 to0.1581, yet worsened matched-
count ADE by0.02047% (nominal locality-bootstrap95% interval[-0.03206,-0.00957]%).
The new allocation avoided less harm than the benefit it removed. This separates
improved probability estimation from useful risk-constrained intervention.

A subsequent registered diagnostic examined1,728 repeated fitting batches from
108 paired fits at initial and final states. In final supervised models, the
shared auxiliary-to-direct-risk gradient norm ratio had median213.257. Total
gradients opposed direct risk in60/432 batches, versus0/432 at initialization.
The diagnostic replayed exactly and did not update parameters or reopen held
outcomes. It motivates a single-factor risk-priority gradient-cap experiment;
it neither proves the cause of the held-policy loss nor validates that repair.

The gradient diagnostic and proposed cap are optimization controls, not claimed
methodological novelty. Raw Euclidean gradients do not characterize finite
AdamW updates or certify selection-conditioned risk. Repeated fitting batches
are not independent sources, and nominal development intervals do not correct
for accumulated research adaptation. The original selected-risk requirement,
independent calibration, confirmation and a positive method comparison remain
outstanding. This evidence belongs in a limitation/ablation discussion, not an
abstract claim of safe world-model success.

Sources: [paired result](../european_easy_hurdle_v1/results.md),
[fitting diagnostic](../european_easy_gradient_diagnostic_v1/results.md),
[optimization prior work](../european_easy_gradient_diagnostic_v1/related_work_note.md).
This note was written after the diagnostic seal and is not a sealed result file.
