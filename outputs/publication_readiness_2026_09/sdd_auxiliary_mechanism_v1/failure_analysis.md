# Why the Tested Source Schedule Is Not a Forecasting Contribution

## Observed Failures

All twelve schedule/input averages lose to CV. All 108 individual fits fail the
easy-degradation limit. Three of the 54 new fits have tiny positive average
gains but are unsafe; these are retained, not described as uniformly negative.
The new fits improve their own training cohorts by 0.535-1.909%, without a safe
held-site benefit. This is a train/held gap, not evidence of inadequate runtime.

Shortening main training from 6k to 4k lowers error in every seed-averaged
input/site cell. Thus the old auxiliary advantage was partly confounded with
main-domain exposure. It was not legitimate to attribute all of it to learned
cross-domain motion. Real labels add no stable benefit over the shuffled-label
control: geometry +0.118%, mask -0.015%, RGB +0.058%, all with descriptive site
intervals crossing zero. This is failure to establish benefit, not an equivalence
test or proof that SDD can never transfer.

## Event-Level Evidence

Event strata are evaluation-only, not inference inputs. They retain the same
11,966 windows: static-stays 177, static-moves 188, moving-stops 279, moving-turns
1,351, other-motion 9,971. Some strata exist in only two physical sites.
Hard rows defined by fold-training CV error quantile total 2,502.

Real-source geometry gives +3.689% on moving-stops and +0.039% on moving-turns,
but -13.625% on other-motion and -0.063% on hard rows. Real-source RGB gives
-4.043% on stops, -1.745% on turns, -20.792% on other-motion and -0.269% on hard.
New control hard averages also remain negative. These subset gains do not
justify an oracle event gate. A deployable gate would have to infer the event
from admitted past inputs and be tested separately.

Static-stays CV error is exactly zero; percentage improvement is undefined.
Real-source geometry/mask/RGB absolute normalized ADE is 0.03164/0.02373/0.03187
on that stratum. Static-moves error barely changes from CV. The networks are not
showing reliable start anticipation, while introducing motion into easy cases.

## Source Support and Normalization

The train-only audit distinguishes broad exact-static-to-any-motion from the
earlier half-box displacement proxy. Broad source support is 10,039 windows and
342 recording-local tracks, but the stricter proxy has only 244 windows and
58 tracks in the same train-40 population. This reconciles with the earlier
census. Neither definition is human gold or a count of independent starts.

A fixed 0.001 native-unit history-scale floor yields very large stationary
source targets. Median broad static-moves target magnitude is 1,125 in SDD,
versus 24.702 in main fold-0 training data. The corresponding analytic scalar
dlog1p(ADE)/dADE medians are 0.000888 and 0.038907. This is not a measurement of
neural parameter gradient norms, physical movement or a proof of causation.
It identifies a representation/loss-sensitivity mismatch worth a controlled test.
The primary normalization and training were not changed after seeing results.

## What Remains Unidentified

- The permutation keeps recording and future-support strata; 197 singleton
  windows cannot move and about 2.4% of donors share the original agent. It
  preserves motion marginals and scene priors, so it is not a fully independent
  null or proof that correct supervision has no information.
- Source warm starts can affect optimization and regularization. The absolute
  ADE decomposition is arithmetic, not identification of a mediation mechanism.
- Low-resolution crops, annotation timing, viewing geometry and temporal-rate
  mismatch remain competing explanations. No missing image semantics are
  manufactured, and raw frame intervals are not equated with seconds.
- Three repeatedly exposed sites cannot establish new-domain safety. Extra seeds
  and overlapping windows do not supply independent scene evidence.

## Next Repair, Not a New Claim

Prioritize a unit-rescaling-invariant internal representation and loss-response
check for stationary source histories, while keeping the existing main
evaluation mapping fixed. Define a source-only or reversible internal transform
from admitted past/train information, test identity and rescaling behavior, then
register one matched comparison. Do not silently change the approved primary.
Measure event-conditioned gradient contributions directly before attributing
failure to loss saturation. More capacity, more source windows or another
threshold sweep is not supported by the current evidence alone.

No model is promoted. Historical Stage26/37 scores are not restored as clean
confirmation results. Stage5C and SMC remain disabled; current results are not
metric, seconds-level, true-3D, foundation or submission-ready evidence.
