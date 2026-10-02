# Predictive Information Without a Safe Decision Rule

## What Worked

Mean per-tree weighted training loss across the 72 heads decreases from 0.683461
to 0.592531 with actual features, versus 0.636319 with the shuffled control.
These are normalized five-moment training losses, not trajectory losses. Both
arms can fit training data, which is why placebo and validation controls matter.

Actual quality features improve validation signed-score MSE in 60/72 heads,
versus 27/72 for placebo. Their mean MSE is lower than original in 11/12 localities
and lower than placebo in 9/12. The registered locality intervals support both
MSE contrasts and all four full/matched utility contrasts. This supports useful
conditional information, not a noise filter or a deployed safety mechanism.

## What Failed

1. Coverage expansion creates additional observed mistakes. Quality selects
   16,978 more repeated occurrences. Known-label easy-risk violations rise from
   4 to 20; nine upper-risk violations have no selected unknown outcomes.
2. Unknown-outcome support also deteriorates. Selected unknown occurrences rise
   918 to 1,143. Of 41 upper violations, 20 already fail using known outcomes and
   21 cross the budget only after unknown-envelope completion.
3. Equal-count ranking is not a complete remedy. Matched utility improves, but
   complete support falls 32 to 17. Known-label violations remain three in each
   matched arm, while completion-only failures rise three to 18. Average useful
   ordering does not establish support for every selected risk denominator.
4. Gains and failures differ by locality. All six 008 and all six 048 heads lose
   complete support despite lower average MSE. All three former 112 violations
   disappear and its support rises to three. This is heterogeneous failure, not
   universal feature uselessness or a single unlucky training seed.

## Two Concrete Cases

The largest quality completion ratio occurs in
`single1_seed43_controller0_dimensionless_fit_eu-locality-067`, head17. Selection
expands from two to 37 occurrences, three unknown. Known easy harm is 0.237375,
selected easy reference mass 17.428977, and unknown envelope mass 179.422730.
Known easy risk is approximately 1.36%, but the conservative upper is 1030.81%.
It is incorrect to describe the entire upper as observed harm. Unknown outcomes
cannot be replaced with zero or filtered using future availability.

The largest known-label quality ratio is in
`single2_seed43_controller0_dimensionless_fit_eu-locality-110`, head29: one
selected occurrence, no unknown, harm 0.460929 over reference 0.352385. Selected
easy harm is 130.80%, although whole-easy degradation upper is only 0.02485%.
This illustrates why overall easy preservation cannot replace the selected-risk
budget. A tiny selected denominator is not an implementation excuse to waive it.

Placebo's worst upper is even larger (199651.50%), driven by unknown envelope
849.408426 over selected easy reference 0.426025. That does not make quality
safe: the preregistered safety comparator remains the original forest.

## Mechanism Not Yet Established

Centered leaf corrections average zero on known training rows before projection,
but can change under a new recording distribution. Nonnegative clipping affected
14,056 raw predicted moment coordinates with quality versus 5,598 with placebo.
These counts are coordinates across repeated readouts, not distinct bad rows;
they do not prove clipping caused the failures. Prediction of benefit, harm,
reference mass and easy-specific moments can improve in pooled squared error
while their decision-relevant ratios remain wrong.

The next diagnostic must isolate those five component corrections on frozen
models and separate added actions from replaced actions. No threshold search,
future-quality inference, relabeling, removing unknown outcomes or selecting a
winning locality/seed is justified. No claim that tracker noise caused this
result. Detailed descriptive data are in `diagnostic_readout.json` and
`risk_decomposition.json`; they are posthoc explanation, not model selection.
