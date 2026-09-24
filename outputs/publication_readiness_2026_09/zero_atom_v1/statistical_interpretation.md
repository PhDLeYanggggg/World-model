# Statistical Interpretation and Claim Limits

## Unit of Evidence

The experiment retains 175756 past-eligible windows, four physical SDD sites and
three training seeds. Windows overlap within tracks/recordings; seeds reuse
sites. Neither windows, sampled training multiplicities, trees nor seeds create
additional independent physical sites. The same four sites have already been
used in research design and failure diagnosis. Excluding a site's rows from a
particular producer is necessary but does not restore untouched confirmation.

Aggregate gains average physical-site percentage improvements over constant
velocity, not pooled pixel errors and not improvement over a selected strongest
baseline. Three seed predictions are averaged for paired site contrasts.
The 3000 paired bootstrap resamples use the four physical sites, conditional on
these fits. There are 18 contrasts with three subsets each, 54 nominal intervals;
they are not simultaneous or multiplicity-adjusted discovery guarantees.
All comparisons, not just favorable ones, belong in the report.

## Three Different Safety Quantities

1. The solver checks its predicted risk budget in original units. This verifies
   arithmetic, not true future risk or prediction calibration.
2. Worst positive-easy site/seed degradation is compared with the unchanged 2%
   ceiling on observed complete labels. An average can hide individual harm.
3. Complete zero-CV cases have a separate exact-zero added-harm requirement.
   The denominator is zero, so folding them into a percentage average cannot
   establish protection. Unknown/incomplete futures cannot establish it either.

The explicit atom uses source leaf frequencies rather than an upper confidence
bound. A probability of zero means no weighted event occurred in the visited
leaves. It does not mean the event is impossible. Shared training rows and trees
also preclude interpreting 128 votes as 128 independent calibration samples.

## Mechanism Controls

Guarded and matched controls select exactly the same number in each recording,
frame and seed. This distinguishes total coverage from which rows are selected.
Matched controls use the original eligibility/risk objective and retain the
guarded feasible incumbent if a numeric proposal violates count, risk or the
predicted objective. Such retained incumbents are not certified optima. Tied
controls may therefore partly reflect shared incumbents, which must be counted.

The prior source zero-CV label was already present in the easy target. A result
here cannot establish that a missing-label bug was repaired. It tests whether a
separate component readout adds predictive value on the existing partitions.
This is not a causal proof that sparse support alone explains every failure.

## Scope

No new neural trajectory predictor, target protocol, easy cutoff, risk tolerance
or threshold search is introduced. Forecasts are cached-verified; readout fits
and evaluation are fresh. There is no independent calibration or confirmation,
no external forecasting result, and no deployment selection. The evidence is
native obs8/pred12 annotation pixels, not historical raw t50, metric physical
safety, seconds-level dynamics, true 3D or foundation-model success. Stage5C and
SMC remain disabled.
