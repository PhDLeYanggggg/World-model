# Why Better Risk Regression Still Does Not Make a Safe Switch

## What Was Tested

I fitted144 unchanged all-risk neural heads, using three controller localities
to fit each head and a fourth to check it. Each comparison retains the same
frozen forecast, causal355-feature schema,2000updates, three seeds and2% screen.
These are fresh training-source diagnostics, not new independent-test results.
The outer readout is unused. The forecaster bank is cached_verified.

## Findings

1. **The heads are not simply unlearned.** Neural reference-cost MSE skill over
   a fitting-only constant is45.37% in fitting sources and13.41% [3.75,22.14] on
   held sources. Harm skill is31.01% and14.88% [10.74,19.01], respectively. The
   top10% predicted-harm rows contain41.91% [37.94,45.79] of held positive harm.
   Average accuracy and ranking have useful signal.
2. **The selected subset is already miscalibrated in fitting sources.** Its
   predicted positive-harm/reference ratio is1.019%, actual2.850% [2.383,3.292].
   Thus an explanation based only on previously unseen sources is insufficient.
3. **Holding out a source makes the problem worse.** Held predicted ratio is
   0.995%, actual4.867% [3.190,6.916]. Ten of twelve locality-mean ratios exceed2%;
   locality020 reaches13.54%. Reference MSE skill drops31.96percentage points
   [21.91,42.05] and harm skill drops16.13points [10.74,22.11] from fitting to held.
4. **Both terms matter.** In held screened rows, actual/predicted reference is
   0.419 [0.352,0.488], while actual/predicted harm is1.804 [1.350,2.284]. The
   reference budget is inflated and positive harm is underestimated. Across all
   held rows the reference ratio is0.984 [0.796,1.167], so an apparently reasonable
   overall mean hides the conditional error. Ratios averaged across localities
   must not be multiplied to reconstruct the aggregate risk ratio.
5. **Damping is not perfectly calibrated either.** Its fitting screen actual
   ratio is1.025% [0.914,1.156], but held ratio2.602% [0.967,5.379]. Four locality
   means exceed2%; locality020 reaches15.79%. Its interval does not support a
   uniform assertion that the average exceeds2%. Easy-protected damping in the
   preceding full-policy experiment is a different population and rule.

## What These Results Do Not Mean

The diagnostic screen uses only the all-risk head, without the old utility,
easy or stationary guards. It cannot be substituted for the complete policy.
Positive harm sums increases and does not subtract benefits; **4.87% is not net
ADE degradation and not easy degradation**. This experiment does not re-estimate
the old policy's trajectory lift. Four zero-reference queries, repeated across
six dependent views per candidate, have no screened positive harm here.

The evidence distinguishes marginal prediction skill, selected-set error and a
source holdout penalty. It does not by itself prove that loss, shared encoder,
unbounded reference output, scale handling or feature support is the sole cause.
The pointwise2% predicted constraint was never a conformal guarantee. There is
no evidence here for a deployable neural advantage, cross-domain confirmation,
physical safety, metric motion or seconds-level prediction.

## Next Controlled Repair

The next minimal contrast should predict the **signed risk-budget excess**,
`positive_harm - 0.02 * CV_error`, instead of assuming separately low MSE on two
moments makes their selection ratio reliable. Retain the same forecasts,
features, seeds, budget and safeguards; compare matched objectives and keep
zero-CV cases. No threshold relaxation or larger architecture is justified yet.
This is an untested hypothesis, not an implemented benefit claim.

Conditional calibration must then use strictly nested source exclusion. In
particular, combining ordinary leave-one-locality-out predictions to calibrate
an outer held locality is invalid if any contributing producer/head was fitted
using that locality's outcomes. Use a separate inner fit/calibration role whose
whole producer chain excludes the outer source. Do not open independent roles
to rescue this development experiment.

## Engineering Finding

A report-only path-join typo stopped the first training-summary render after
the numeric results had been written. It was repaired and regression-tested.
No targets, features, model weights, thresholds or numeric evaluation changed.
The forecaster/risk failure is not attributed to that reporting error.
