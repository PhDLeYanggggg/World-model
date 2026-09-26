# Factor Attribution: Neither Factor Alone Repairs the Estimator

## Material Passport

Fresh label-assisted diagnostics from cached_verified checkpoints and source
arrays; 36 group readouts / 144 held-locality folds. No new fit, threshold or
policy evaluation. Registration 165bc0a4 preceded the attribution readout.
The parent failed cost gate remains failed. Independent roles stay closed.

## Main Finding

Knowing realized easy membership is insufficient. Replacing p by true E in
E*m produces zero positive, one negative and five overlapping full-input
easy-harm MSE intervals against original_mean. Assignment gains range -172.45%
to +8.53%. Relative to the current composition, this label-assisted substitution
has one positive, one negative and four overlapping intervals. Thus the prior
outside-easy excess finding does not justify a probability-only repair.

Replacing severity m by realized H also fails: p*H has zero positive, two
negative and four overlapping intervals versus original, with all six points
negative (-136.80% to -13.04%). Both substitutions use labels unavailable at
inference. They are not deployable models, achievable bounds or success gates.

## Why the Earlier Error Story Was Incomplete

Outside-easy rows dominated the INCREASE over the original model in 50/53
worsening full views. That is different from dominating TOTAL current MSE.
In the new attribution, outside-easy absolute MSE shares across assignments
are only 6.36% to 31.27%; most total error remains inside the easy event.

Membership-squared terms divided by composed MSE range 0.299 to 2.025;
severity terms range 0.876 to 2.662. The signed cross term ranges -3.687 to
-0.189, with five negative intervals. Errors partly cancel. Setting one
factor to its realized label can remove that cancellation and worsen the
prediction. Ratios over one are legitimate here, not probability estimates.

The unweighted membership Brier assignment points are 0.0734 to 0.1256;
severity-weighted values are 0.0233 to 0.0794. Those weights change the
population. Smaller weighted Brier does not prove better calibration. A
single realized binary E also includes uncertainty that is not attributable
solely to classifier miscalibration. This is an algebraic diagnosis, not a
causal estimate of recoverable error.

## Repair Selected for the Next Controlled Test

Retain direct nested cost prediction and the original four moment losses.
Test easy-membership supervision as an auxiliary representation task, with
an independent output that is NOT multiplied into the harm estimate. Compare
an otherwise identical cost-only head against cost-plus-membership training,
and require improvement against the original model as well. Keep the new
capacity, source split, sampled rows and update budget matched. Do not use
the failed factorization to select another intervention threshold.

This is a proposed controlled training repair, not an observed model gain in
this diagnostic. Its results must be reported separately after fitting.

## Limits

Three seeds, 3,000 resamples of four localities per assignment. Roles/windows
are repeated, historically exposed source development and not independent or
multiplicity-adjusted. Full/motion-only disagreement populations differ.
Detector-derived pixels and annotation steps only. No human-gold, metric,
seconds, physical-safety, true3D/foundation or submission claim. No change to
the approved endpoint or 2% tolerance. Stage5C/SMC remain off.
