# Failure Analysis

## Confirmed Implementation Finding
The old cap-event auxiliary inherited an easy-membership initial probability.
This experiment isolates that mismatch: only membership.bias changes. All288
heads pass matched-initialization/sampler/target checks. No unknown labels were
sampled.1440 source-held MSE checks agree with direct arithmetic; old metrics
recompute exactly. Runtime failure, row misalignment and changed budgets do
not explain this experiment's mixed outcome.

## Partial Optimization Benefit
Against the old true auxiliary, full-family fitting easy-harm cost has four
positive assignment intervals and no negative interval. This is evidence of
an optimization effect under the registered budget. It is not proof that
initialization was the sole failure mechanism: one fitting interval versus
cost-only is still negative, and the source-held magnitude gate fails.

## Ranking Does Not Determine Expected Severity
Full-family source-held harm-presence AUROC improves versus cost-only in all
six comparisons, but primary cost MSE improves in only one interval and
worsens in two. The ranking uses the predicted easy-harm moment to separate
H_easy>0 from zero, not the auxiliary cap-event probability. It is therefore
possible to rank risky rows better while assigning the wrong severity.
Top10 capture and coverage improve in some contrasts; all-harm MSE remains
an adverse guard. No conclusion that every ranking/tail measure failed is
warranted, and none of these secondary gains can override the primary gate.

## What Remains Unresolved
The result does not identify whether the remaining problem is conditional
severity estimation, producer/locality transport, feature insufficiency, or
unsupported tails. Four localities per assignment and overlapping development
roles limit interpretation. The full/motion comparison is not a clean feature
ablation because forecasts and populations also change. The experiment does
not show an independent-domain or trajectory-level gain.

## Consequence
Do not deploy the repaired auxiliary or select its sole favorable assignment.
Do not rerun the same failed magnitude claim under a new label. Existing
fractional, frozen-readout, membership-composition, severity-weighting and
gradient-projection negatives remain relevant. A new hypothesis must separate
ranking from severity with honest fitting-only out-of-fold predictions and
the same strong control. No calibration on the source-held readout is allowed.

Pixel/native-step and silver-label boundaries remain. No Stage5C or SMC.
