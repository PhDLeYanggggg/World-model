# Calibrating Forecast Amplitude Without Changing Switches

The fixed quarter-amplitude development control increased original
recording passes without changing switches. Test whether a source-fitted
model can improve on it without merely reducing the amplitude budget.
Use the same 34 exposed recordings, 14 source groups, three forecast
seeds, CV reference, easy/hard cuts, original 2% rules and candidate
switches. There is no independent or external evaluation.

For each original switch, fit a 32-wide model for available-label
lower utility and positive all/easy harm, including the original
unknown radius. Predict costs for coefficients 1/8, 1/4, 3/8, 1/2
and 1. Use a fitting-only normalized Smooth L1 objective and 220
steps at batch 128. Use seven fitting sources, six donor calibration
sources, and one held source in each of the existing folds. Upstream
features for the fitting switch roster are source-excluded. No
future positions or recording totals are features.

The primary contrast is donor-calibrated row-adaptive versus the
fixed quarter policy. The row score is predicted utility minus a
nonnegative price times predicted all/easy positive harm. Choose
the price on donor recordings only. The donor-calibrated uniform
control uses the same gate, but one amplitude across all switches.
A qualifying setting adds at least one pass, loses no prior passes,
adds no signed-easy violation and has nonnegative paired lower
utility against quarter on the donor set. If none qualifies, use
quarter. This is an empirical donor selection rule, not a coverage
guarantee and not a way to pass absent references.

As a separate diagnostic, match both the switch roster and the sum
of interpolation coefficients per query. For a single switch, use
quarter. For two, compare the pair (1/4, 1/4) with either ordering
of (1/8, 3/8), scoring with the same model and fixed price 1.
This matches a coefficient budget, not a pixel-distance budget.
It is not the operating policy. Use original scoring, reference
support and missingness charges; report source bootstrap intervals
after averaging seeds. No deployment, Stage5C or SMC.
