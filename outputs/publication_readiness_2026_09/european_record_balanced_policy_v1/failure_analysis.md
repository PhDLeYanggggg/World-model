# Failure Analysis

This is a post-fit analysis of frozen TRAIN results. It selects no new model,
threshold, checkpoint or test subset. All prior negative results are retained.

|Hypothesis or question|Observed evidence|Conclusion|
|---|---|---|
|Are small-record constraints diluted by utility weights?|Uniform constraint weighting increases supported recording views84to131; the increased-weight slice improves30to76of372.|Partial mechanism support, not a complete safety repair.|
|Is improved coverage caused by discarding almost all benefit?|Eligible positive benefit retention81.5516%to80.9251%; selected occurrences293,781to292,961.|Most benefit is retained; this is not simply a near-empty policy.|
|Does average utility improve?|Full utility completion bound relative to parent is[-0.135714,-0.133642]pp.|No. The matched-count improvement is small and does not reverse the full-policy tradeoff.|
|Do all three head seeds improve recording coverage?|30to46,27to40,27to45of148.|Yes on fitting rows; fixed forecast seed and repeated scenes prevent an independent-generalization interpretation.|
|Are failures just missing labels?|289known all-harm violations remain; upper-bound violations290.|No. Known harmful decisions dominate the all-risk failure.|
|Does net easy preservation establish safety?|Zero net easy-degradation violations, but39known and59upper easy-positive-harm violations.|No. Both criteria remain required and distinct.|
|Did aggregate and record-level criteria move together?|Aggregate coverage54to52of72; record coverage84to131of444.|No. Averages conceal different within-source behavior.|
|Can the penalty simply be increased everywhere?|Actual coefficient factors range0.114286to28.499130; larger recordings can receive less weight.|This experiment redistributes penalties. It does not establish that a global multiplier solves the problem.|
|Is the all-risk dual cap preventing feasibility?|No all-risk dual reaches100; maximum17.9144;286positive excesses before the final update.|No direct evidence for a cap-only repair. Dual rate, optimizer convergence and representation remain unresolved.|
|Is the entire candidate set feasible?|The prior exact TRAIN diagnostic lacks a supported easy-reference action forlocality082/51in two contexts.|A selector-only repair cannot certify the whole roster with unchanged candidates.|
|Where are violations concentrated?|125/168locality008 and160/192locality074known all-harm violations.|285of289remain in the two multi-recording sources. Keep them in every subsequent evaluation.|

## What Changed and What Did Not

59 recording/head views gain a pass,12lose it,72keep it and301remain failing.
The favorable total does not erase individual regressions. Record-balanced
known benefit is15.911456pp, known harm0.558790pp and unknown envelope0.000693pp
of macro-averaged reference cost. The parent values are16.034646,0.547302and
0.000926pp. Fewer violating recordings do not imply a lower total harm mass.

The record IDs are training constraint groups only, not newly added inference
features. All features, forecasts and normalizers are fixed. Future labels enter
losses and diagnostics, never the inference interface. No new goal construction,
central velocity, test normalization or independent-role access occurs here.

## Next Repair Must Address Two Distinct Problems

1. **Candidate support:** inspect the source-TRAIN forecast errors behind the
   known unsupported recording and train/evaluate a legal forecasting repair.
   Keep candidate definitions frozen before scoring. Avoid treating a baseline
   copy or negligible intervention as new neural support to game a ratio.
2. **Policy fitting:** test constraint convergence or benefit/harm representation
   while retaining this cost-sensitive reference. Do not change architecture,
   penalties, data and thresholds simultaneously. An exact label-aware safe set
   is evidence of available choices, not evidence that causal inputs can identify it.

Neither repair has been run in this experiment. Independent generalization,
scene-joint utility and calibrated deployment remain unproved. The proposed
weighting is an optimization diagnostic, not a new constrained-learning theorem.
The project remains a 2.5D top-down research track, not submission-ready or a
foundation world model. Stage5C and SMC remain disabled.
