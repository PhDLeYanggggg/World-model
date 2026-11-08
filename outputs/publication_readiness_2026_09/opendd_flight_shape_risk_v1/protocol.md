# Recording-Risk Training at Matched Forecast Radius

The prior shape learner improved trajectory utility at identical
interventions but lost 34 passing recording-seed views and recovered
four. Every loss crossed the easy upper limit despite the same
reference mass and unknown-label debit. Test whether optimizing the
recording loss directly can retain the shape gain without this trade-off.

Use the same exposed OpenDD development cohort, seven held locations,
three seeds and the same pair-excluded incumbent producers. Keep exact
incumbent recipients and maximum departure from CV. Continue the
self-only shape parameterization; no neighbor input or selection on
held results. Start all new heads at the incumbent. A training update
chooses a donor location uniformly, then one of its edited recordings
uniformly, and consumes every eligible target in that recording. This
keeps the reference masses exact, rather than estimating a ratio inside
a row minibatch. Use 600 updates and identical draws for all arms.

1. \`flight_point\`: previous pointwise easy-harm objective on the new
   matched recording sampler.
2. \`flight_budget\`: add a hinge on the exact original all-upper,
   easy-upper and signed full-easy 2% ratios. The CV reference and
   fixed unknown debit are included. Skip the undefined terms only
   in training, and count undefined recordings as failed at readout.
3. \`flight_relative\` (primary): add to arm 2 the positive increase of
   recording easy harm over the incumbent, divided by its CV easy mass.
4. \`flight_noncancel\`: replace arm 3's recording-level relative
   hinge with the sum of positive per-target increments. This does not
   let improvements on one target subsidize additional harm on another.

The budget and relative terms have fixed weight 8, unselected on held
outcomes. Calculate them in release coordinate units; the base utility
loss retains the existing per-target causal scale. Keep the frozen
incumbent, hindcast direction, and previous pointwise easy-guard model
as controls. The latter is a hash-verified readout, not a fresh fit.

Report the paired conservative utility interval (3,000 location draws
after averaging the three seeds) and all original recording risks.
The primary question is whether arm 3 both improves on the incumbent
(lower CI above zero) and passes at least as many original risk views.
This diagnostic recovery criterion is separate from the unchanged
requirement to pass all recording views before deployment. Report
gain/loss pass transitions, undefined views, and relative pair error.
No change of split, threshold, output support, or risk definition.
No external or DroneCrowd outcome access, new deployment, Stage5C or SMC.
