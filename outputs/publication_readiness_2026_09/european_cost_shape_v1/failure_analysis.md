# Failure Analysis

## Material Passport

Fresh source-development experiment on frozen neural assets. This analysis
separates numerical findings,descriptive mechanisms and unresolved hypotheses.
It does not select a replacement policy from held results.

## 1. Origin-Bound Infeasibility Was Not the Whole Problem

The parent left99/432 easy-mass fits unmatched at its fixed slope bound.
The new constant-capacity option and monotone shapes match both means in all
432 constrained readouts. Numerical certificates pass. Nevertheless,full
cost-only shape mass has no positive primary interval against raw or L2.
Removing the feasibility defect is therefore insufficient for this repair.

## 2. Fitting Mass and Held Squared Error Still Conflict

Shape mass versus shape L2 uses the same knots,fractions and sequential
architecture. Adding mean constraints produces five negative primary
intervals and one overlap in each of the three full-input arms. Coverage can
improve without accurate allocation of harm across individuals or scenes.
The all-harm fit changes the easy-harm cap,so this is a sequential readout
comparison,not a jointly optimized two-component ablation.

This does not refute squared error as a mean-estimation objective. Restricted
finite estimators,limited scene support,nested constraints and transport may
all matter. A global empirical moment is weaker than a conditional mean.

## 3. Better Fitting Does Not Reliably Transport

Against origin mass,full-input cost-only shape mass improves fitting MSE in
66/72 views,but31 of those do not improve held MSE. Against origin L2,the
numbers are15 fitting improvements,11 of which do not improve held MSE.
See fitting_transport.md for all72 comparison/component summaries,including
motion-only. Fitting weights are equal-locality;held MSE is row-weighted
within its locality. These are dependent view counts,not independent trials.

This supports a transport problem beyond the particular slope bound. It
does not uniquely establish missing images,missing interaction,rare-event
sample size,label-cut shift or inherent unpredictability as the cause.
Those alternatives require their own controlled tests.

## 4. Reallocation Damages Ranking and Tail Capture

Full-input cost-only shape mass versus raw worsens positive-envelope harm
AUROC in all six intervals and top10 capture in four. Versus origin L2,
AUROC again worsens in six and top10 capture in three. The transformations
are monotone in normalized fractions,but multiplying by row-specific envelopes
and corrected caps can change ordering of absolute risk across rows.
Monotone fractions do not guarantee preserved absolute-risk ranking.

## 5. Relative Percentages Need Absolute Context

Full-input cost-only held easy-harm MSE medians across24 dependent locality
views are0.041005 for origin L2,0.040793 for shape L2,and0.044081 for shape mass.
These medians are descriptive and do not replace the registered paired means.
Shape mass's mean-of-seed coverage ranges from0.0391 to12.188 despite matching
every fitting mean.

Motion-only failures are retained. In P0/C1,locality110,the three-seed mean
MSE changes from0.0000102348 with origin L2 to0.0008543204 with shape mass.
Its mean-of-seed coverage is1593.19. Small denominators amplify percentages,
but the absolute overprediction is real. The registered motion-only primary
point range against L2 reaches-27619.23%;the ratio of seed-averaged MSE is
not the same statistic as the average of seed-specific percentage gains.
No figure clips these negative intervals or substitutes a different metric.

## 6. Auxiliary Signals Are Not a General Repair

True auxiliary shape mass has1/6 positive primary intervals against cost-only
and2/6 against shuffled. Other intervals overlap zero;the point ranges include
negative values. There is no six-comparison primary/guard success. These
signals remain development observations,not independently confirmed gains.

## Consequence

Stop generic global-readout tuning. Keep forecasts and deployment unchanged.
Check whether available past context and effective event support can separate
same-score rows with different harm,using only fitting-side nested data.
Do not infer a universal impossibility result from this finite negative study.
Independent roles remain closed;Stage5C and SMC remain off.
