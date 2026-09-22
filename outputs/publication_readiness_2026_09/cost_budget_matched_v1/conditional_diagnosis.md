# Where Conditional Harm Underprediction Begins

## Material Passport

Fresh post-readout diagnosis of36 cached-verified equal-budget heads. No model,
choice, threshold, target, role or inference feature changes. The analysis
contains2016 records across12views, two populations and12fixed causal choice
groups, with all/easy/training-defined disagreement strata. Easy is a future-
defined diagnostic only. Cost means use complete labels; unknowns remain unknown.
All four sites are research-design exposed. No independent confirmation.

## Main Finding

| Population and subset | Native views underestimating harm | Fraction | Intermediate |
|---|---:|---:|---:|
|Fitting, all causally eligible rows|1/12|0/12|1/12|
|Fitting, each head's own strict selections|12/12|5/12|8/12|
|Held source, all causally eligible rows|4/12|5/12|5/12|
|Held source, each head's own strict selections|12/12|12/12|12/12|

Global harm estimates are generally conservative in fitting, but selecting on
those same estimates exposes optimistic conditional errors. The issue is not
a single global multiplicative offset. Held scenes worsen the conditional gap.
The analysis cannot identify feature insufficiency or irreducible uncertainty
as the cause; it establishes where the error is visible.

On the SAME intermediate-selected fitting population, native and fraction
underestimate harm in0/12views, versus8/12for the selecting intermediate head.
On the held intermediate-selected population the counts are5/12,3/12,12/12.
Comparing different selected populations alone would hide this distinction.
This is descriptive selection-conditioned evidence, not a causal proof of
winner's-curse bias or a calibrated ensemble guarantee.

For native's own fitting selections, realized/predicted mean harm ranges1.42-3.62;
on held selections it ranges2.58-6.42. Intermediate ranges0.68-1.64 in fitting
and2.06-3.78 held. Fraction ranges0.74-1.55 in fitting and1.38-3.41 held.

## Easy Failure Is Not Removed

On deathCircle's complete intermediate-selected positive-easy rows, the three
seeds have77/102/144 supported rows. Realized harm means are2.173/1.636/1.792px,
versus intermediate estimates.170/.127/.111px. Net gains are negative in all
three: -.948/-.385/-.665px. The other objectives are less optimistic on that
same population but still underestimate its harm. These complete-only numbers
do not replace the full available-label easy-degradation scores or resolve
incomplete outcomes.

## Fixed Repair to Test

The next registered experiment changes fitting allocation only: fourfold
loss emphasis on the frozen intermediate head's causal fitting selections,
renormalized under the original scene-uniform sampler. Architecture, target,
training budget, draws, prediction features and decision thresholds stay fixed.
Both benefit and harm errors receive the same weight. This tests a finite-
capacity fitting hypothesis; it is not a new calibration method or a guarantee.
The weighting reference is fitted on these training labels, so its predictions
are explicitly in-sample fitting tools, not held-out calibration scores.

The repair must improve held forecasting AND protection. Better fit on the
old region alone is insufficient, especially if the new selected region becomes
optimistic again. Compare both regions, preserve all seed/site failures and
keep the independent scene-calibration gap explicit. No multiplier sweep or
threshold repair on held outcomes follows failure.

## Verification

Five diagnostic tests check aligned common populations, harm frequency-times-
severity identity, empty/unknown support, causal selection invariance under
future-label changes and invalid-input rejection. Parent checkpoints, archives,
labels and source bindings are verified by the existing matched-budget loader.
The2016-record output SHA is
`4917cd97cf25b66d221231473103411edf7a811f8c2a5d8292ac066018737459`.
No deployment, Stage5C, SMC, metric, seconds or foundation-model claim.
