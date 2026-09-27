# Fixed-Floor Harm Tail Weighting

## Material Passport

Registered source-development experiment. Fresh_run:216 neural risk-head fits,
held inference, fixed decisions, query-count matched comparison and3,000 paired
locality-bootstrap draws. Cached_verified: nine forecasting banks, the complete
protected-damping producer chain,108 floor-target ridge utility models and their
training-only preprocessing. Independent roles, new trajectory training and
deployment are not_run. Same12 opened source localities, obs8/pred12 rawstride12.

## Question and Matched Change

The prior fixed-floor linear screen obtains incremental ADE benefit but
underestimates selected positive harm. Harm output clipping to zero and reference
overprediction are documented. We test bounded nonnegative neural risk heads,
changing ONLY the loss weighting between the two new arms. Neither arm clips
an unconstrained negative harm prediction to zero.

The same four-source forecaster and four-source floor controller are frozen.
The remaining four sources rotate through all six two-fitting/two-held splits.
Both new arms have identical380 causal features, initialization, draws,2,000
updates, optimizer, width64, clip10 input standardization and cost scale. The
predecessor's floor-target utility score, movement guard and99% input-support
rule stay frozen for BOTH arms. Future labels/masks do not enter inference.

Outputs: all-reference cost, bounded all-harm, easy-reference cost and bounded
easy-harm. Reference outputs use softplus; harm outputs are sigmoid fractions
of the same past-only max rollout disagreement envelope. Targets are relative
to the same frozen floor on fitting and held rows. Easy event retains the
producer-training CV cut. Unknown-label rows never enter supervised draws.

Control: unweighted moment MSE. Treatment: for each harm output separately,
compute the equal-source weighted90th percentile among POSITIVE fitting harm
labels, multiply loss by4 at/above that threshold and by1 otherwise; normalize
weights by their fitting weighted mean. Reference losses stay unweighted.
If a harm output has no positive fitting labels, retain unit weights. No held
labels set these weights; no multiplier/quantile search. This differs from the
older dynamic underprediction penalty. Weighted scores are tilted moments,
NOT automatically calibrated expected harm or upper confidence bounds.

## Fixed Decisions and Equal-Coverage Control

The same positive frozen utility, movement and support guard precedes each
arm's all/easy predicted2% risk screens. Rejected actions retain protected
damping. Also retain predecessor ridge screen and no-intervention floor.

For every SAME recording/frame query, let K be the treatment's selected count.
Select exactly K eligible agents using the control score
`max(all_harm-.02*all_reference, easy_harm-.02*easy_reference)` (ascending),
breaking ties by fixed row ID. Both decisions see only current-query causal
scores, never other time frames or outcomes. This matched-count control may
violate its predicted budget and is diagnostic, not a deployed rule. Include
all agents, including unknown-label targets, when matching counts.

## Evaluation and Decision

Primary: difference in selected positive-harm/floor-reference ratio between
MSE matched-count control and treatment (positive means treatment reduces harm).
If any registered source has no defined selected denominator, keep the primary
undefined and do not call full fallback a successful repair. Success also
requires nonzero intervention per locality, no observed source/view risk above2%,
easy preservation within2%, no zero-CV harm, and useful net ADE beyond the floor.
Report ordinary MSE/treatment differences AND the matched-count ADE contrast;
count reduction alone cannot establish better ranking.

Report all/easy/hard ADE, endpoint FDE, complete/partial support, worst locality,
tail error, intervention, unknown-label intervention, selected harm, cost MSE,
predicted/realized harm and reference moments, three seed views and training
losses. Average dependent views within12 localities then bootstrap3,000 times,
seed91531. All readouts are development-exposed, not confirmatory/simultaneous.

Register before100-update pilot. Resume pilot into the same2,000-update budget.
Freeze all216 score banks and108 action groups before outcome scoring. Save
atomic checkpoints each500updates, RNG/optimizer/draws and live heartbeat.
CPU4/interop1, workers0, native arm64 Python; keep10GiB disk. Training smoke
estimates whether local execution is reasonable; no slowdown-only downgrade.

## Claim Boundary

No independent calibration is claimed in this loss-isolation experiment.
Empirical screening and a bootstrap do not certify safety. Only the predictive
risk heads are retrained, not the trajectory model or JEPA encoder. Released
detector silver and image-local raw-frame results are not human gold, metric,
seconds, historical Stage37 recertification, physical safety, true3D, foundation
or submission-ready evidence. Stage5C and SMC remain disabled.
