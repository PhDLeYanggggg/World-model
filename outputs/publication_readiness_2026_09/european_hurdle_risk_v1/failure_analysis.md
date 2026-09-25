# Failure Analysis: Better Occurrence Scores, Unresolved Intervention Risk

Result source: fresh training/readout plus explicitly posthoc diagnostics on
the same opened source localities. Nothing here is an independent final test.

## Findings and Causal Limits

| Explanation | Evidence | Conclusion |
|---|---|---|
| No positive harm examples for fitting | None of 144 fitting-locality/event slices is empty | Not supported as a general explanation; easy-event support is sparse, not absent |
| Moment MSE does not identify occurrence and severity separately | Equal-architecture hurdle supervision improves positive-easy protection and several occurrence scores | Partial support for changing the loss; not proof of uniquely identified conditional risks |
| Better risk learning always improves accuracy | All-event neural all-ADE improves in 8/9 matched comparisons, but easy-event all-ADE worsens in 9/9 | False in this experiment; event-specific tradeoff remains |
| The repair protects zero-reference cases | All 12 hurdle views containing such rows add error to some of them | Failed observed protection; aggregate easy averages conceal this stratum |
| Zero-reference support generalizes | Only four distinct rows in one locality; two fitting folds have none | Insufficient support; six nominal safety passes contain no zero-reference readout cases |
| Neural motion is more useful than matched damping | No positive all-ADE interval; 15/18 all and 15/18 hard intervals favor damping | No stable neural dynamics advantage |
| Hard-subset benefit is absent everywhere | Two fold-2 easy-event comparisons have positive hard intervals | Narrow positive signal, not a consistent cross-fold result |

## What the Zero-Reference Audit Establishes

The four zero-CV-error rows all belong to `eu-locality-008`. Folds 0 and 1 exclude
that locality throughout the producer chain and contain no zero-reference
fitting examples. Fold 2 includes the locality in fitting and has none in
readout. The twelve hurdle views from folds 0/1 harm one to three of the four
rows. Added ADE ranges 0.05294-0.92109 image pixels. The zero check is separate
from the positive-easy ratio, whose denominator excludes zero-reference rows.

This is a measured support gap, not a demonstrated sole cause. Damping's new
heads protect those same rows despite the same fitting gap. Candidate geometry,
estimated reference mass and selected-tail uncertainty may also matter. We did
not change the zero definition, fit a readout-derived guard or exclude the rows.
Zero CV error is a future-defined evaluation condition, never a legal input.

## What the Probability Check Establishes

Occurrence probability is supervised for joint event-and-harm, not for harm
conditional on a future-known easy label. A posthoc fixed fitting-prior control
gives positive equal-locality Brier gains in 9/9 neural easy-event views (8/9
positive intervals), and 6/9 all-event views (6/9 positive intervals). This
supports learnable occurrence signal, not independent calibration or reliable
ranking of interventions. Product-MSE factors alone are not identifiable as
probabilities, so superiority to their Brier score would be a weak comparator.

Hurdle heads still retain the old utility head, a direct reference-mass moment,
and the same pointwise risk ratio. Any remaining gain/harm ranking error or
reference-mass overestimate can affect selection. Population reliability does
not imply reliability on the selected tail. Intervention rates change, so the
current experiment does not isolate ranking quality at matched coverage.

## Next Falsifiable Step

1. On opened development sources only, compare risk rankings at fixed matched
   intervention counts, preserving both candidate families and both event
   definitions. Separate improved ordering from merely fewer interventions.
2. Characterize causal support for near-zero-reference behavior using fitting
   observations only. Do not train a guard from the four readout failures or
   claim that a constant-motion history proves a future will remain constant.
3. If a changed score or support rule is justified, register the complete
   producer-excluded calibration comparison before fitting it. Keep the 2%
   criterion and reserved confirmation roles unchanged.

These are next actions, not completed experiments. Larger trajectory models,
more thresholds and a selected favorable seed are not supported repairs yet.
Deployment remains unchanged. Historical Stage35/37 outcomes remain exploratory,
not an independently recertified safety floor.

All uncertainty is conditional on twelve opened European Squares localities;
shared windows/views and posthoc subgroup inspection limit the claims. This is
image-pixel 8/12 detector-track forecasting, not metric or seconds-level physics,
human-gold annotation, true 3D, foundation modeling or physical safety. No
Stage5C or SMC was executed.
