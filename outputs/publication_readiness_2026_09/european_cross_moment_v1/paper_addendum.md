# Target-Definition Ablation for Risk-Controlled Forecast Intervention

## Method

We distinguish ordering realized harm shares from ordering conditional risk
moments. The auxiliary uses signed cross costs H_i B_j-H_j B_i with a logistic
pairwise loss over predicted log moment ratios. Two sequential controls isolate
the cross-cost weighting and the use of a fixed fitting-only loss normalizer.
The construction is motivated by a conditional sign identity under independence,
not offered as a consistency theorem for overlapping trajectory observations.
All forecast producers, inputs, optimization budgets and the intervention risk
limit are held fixed. The study contains 72 fresh heads and three training seeds.

## Results

Cross weighting improves damping/all ordering in all nine fold-seed comparisons
at both matched intervention-count anchors; full-policy changes are +0.5105 to
+1.1595 pp of CV-normalized ADE gain. The same change does not consistently help
neural candidates and worsens damping/easy full-policy performance. Fixed
normalization partly improves easy-event ordering but is not a universal repair.
All 18 neural-versus-protected-damping all-ADE and hard-ADE points remain negative
under each new mode. Positive-easy checks pass, but observed zero-reference harm
remains. The complete matrix, not a favorable arm, is the reported evidence.

## Claim Boundary

This supports a candidate-dependent controller-target effect, not stronger
neural world dynamics. All twelve localities are opened development, with
complete upstream exclusion only relative to each fit. Three-seed comparisons
and 3,000 locality bootstrap draws are conditional and dependent, not independent
confirmation or multiplicity-adjusted population evidence. Four zero-CV cases
have only two future labels each; they cannot validate full-horizon protection.

No architecture novelty or calibrated-safety theorem is claimed. Stronger
contributions require candidate/controller disentanglement, reliable annotation
support, external confirmation and a literature comparison. These are image-
pixel raw-stride 8/12 experiments, not seconds, metric, human gold, true 3D or
foundation evidence. No model promotion, Stage5C execution or SMC.
