# Nested Selected-Moment Calibration

On the unchanged A development bank, compare the exact filter-first control
with calibrating all/easy positive-harm numerators, selected-reference
denominators, or both. The original 2% evaluation and risk supports remain
unchanged. No new candidate trajectory, feature, risk threshold or data role.

For each target A source and actual forecast seed, predict each of its 13
calibration donors using an inner cost model fitted on the other 12 A
sources. Neither the target nor the donor can contribute to that inner fit.
Apply the exact raw filter on the donor. Sum predicted and true components
on its selected rows per recording, including the causal disagreement
bound on wholly unknown rows. Normalize underpredicted harm and overpredicted
reference residuals by the sum of path radii. Keep zero actual reference
recordings in residual accounting. For each component take the higher 90th
percentile of within-source maxima, clipped to a nonnegative adjustment.
Fewer than three donor sources with selected actions disables calibration.

Add harm adjustments and 2%-weighted reference adjustments to the predicted
excess of each candidate, scaled by its causal radius. Keep raw utility and
query cap. The target's final original ridge model still fits 13 sources
versus 12 in the inner costs: this small training-size mismatch is not a
conformal guarantee. Reranking within screened actions can change which
paths are picked. The calibration is empirical, not a safety certificate.

Freeze all decisions for the three seed banks before scoring. Primary arm:
joint harm and reference calibration versus raw. The two component arms
are diagnostic. Report all 102 views, actual switches, known/completion
easy violations, undefined references, paired utility versus raw and CV,
3,000 source bootstrap draws after seed averaging, and exact raw replay.
Empty decisions remain unsupported. No external label access, physical
safety claim, deployment change, Stage5C or SMC.
