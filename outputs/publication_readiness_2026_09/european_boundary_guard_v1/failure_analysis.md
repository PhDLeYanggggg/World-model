# Better Observed Routing, Worse Required Risk

## Completed Experiment

All 36 real Torch critics, two exact full-budget refits, 72 original policy
groups and separate numerical replay are complete. Full new cost inference,
saved decisions and all original full/record control scores match exactly.
The two arms share capacity, initial weights, causal inputs, complete sampling
and budgets. This is exposed SOURCE fitting, not an independent test.

## What Worked

The added boundary loss changes decisions substantially. Known-label ADE gain
over the same floor rises from 10.578407% to 21.425949%. The paired known effect
is +10.847543 pp, conditional locality-bootstrap 95% CI [9.699275, 11.991599].
Every one of the 12 locality-averaged known effects is positive. Relative to
the original frozen guarded TDEF, the known effect is +9.535749 pp
[8.128308, 10.898344]. These are useful observed routing results, not safety
or independent generalization claims.

Safe easy action occurrences rejected by the predicted guard fall from 498,455
to 144,757; rejected strictly beneficial safe easy occurrences fall from
466,569 to 121,127. Pointwise reference-support blocked views fall from 9 to 5.
The loss therefore does address part of the preceding false-rejection problem.
These are repeated candidate/context/seed occurrences, not independent samples
or counts of actual chosen switches.

## What Failed

|Quantity|Moment continuation|Boundary supervision|
|---|---:|---:|
|Actual switch rate, repeated row views|32.979176%|70.412236%|
|Known unsafe action occurrences admitted|173,754|563,612|
|Required recording-risk views supported|20/444|0/444|
|Known all/easy risk violations|121/240|388/417|
|Undefined selected-easy views|17|6|
|Worst known easy net degradation|5.769174%|40.472291%|
|Known easy views above 2%|8|79|

The new critic recovers support by admitting many more actions, including
many harmful ones. Five repeated views retain a singleton-query known-support
obstruction. A better ranking or another actor cannot repair that unchanged
eligibility constraint. Conversely, reducing this support count does not
establish the safety of the other recordings.

The primary completion-lower effect is +4.830476 pp, but CI
[-3.072394, 9.335094] crosses zero. The analogous effect over the frozen guard
is +4.727428 pp [-1.495463, 8.613226]. Both fail the registered criterion.
Locality082 has known +7.615515 pp versus completion-lower -35.809462 pp;
it remains in the full comparison. This difference is missing-outcome
uncertainty, not measured hidden harm. Observed all/easy violations already
make the policy unsafe under the stated error-budget contract, independently
of those completion bounds. Dropping that locality would not repair safety.

## Loss And Mechanism

The same fixed-query anchor objective ends at 0.311329 for the moment arm and
0.453420 for the boundary arm, from the common initial 0.516649. Boundary BCE
ends at 0.892041 and 0.435365 respectively, from the common initial 0.979314.
The auxiliary succeeds at its fitting objective while worsening the anchor
fit and complete policy risk. These are training monitors, not validation loss.

The [method note](method_scope.md) gives a precise limitation of this repair:
an event probability is not an expected-harm budget. Tying its logit directly
to a mean-cost ratio generally changes the population mean-cost optimum.
Detaching the denominator prevents a direct reference-inflation gradient but
does not repair that semantic mismatch. Shared features can also alter benefit
and reference estimates indirectly. The elementary argument and measured
tradeoff do not uniquely attribute every finite-network failure to one gradient.

Neural action use is 99,511 versus 99,706 repeated rows, approximately 11.1%
in both arms, despite the large change in total switching. Equal counts do not
imply the same chosen rows. No path-use ablation establishes a neural forecasting
increment, and no trajectory model is retrained. Do not turn the routing gain
into a JEPA, Transformer or world-dynamics contribution.

## Next Repair

Do not tune this readout's thresholds, extend the same tied objective merely
because BCE is falling, or call its sigmoid calibrated confidence. Preserve
the original 2% requirements and failed results. The next isolated control
should separate event prediction from proper cost magnitude semantics or test
a proper conditional-moment factorization, after checking existing native-
protection and hurdle models to avoid repeating them. It still needs a full
policy test, including support, positive harm and missing outcomes.

Fitting gates remain 5/9. Deployment is unchanged; no independent role or
reserved DroneCrowd data is opened. These local-image detector-silver
obs8/pred12/stride12 raw-frame results are not SDDt+50, metric/seconds, human
gold, true3D, foundation, physical safety or submission-ready evidence.
Stage5C and SMC remain off.
