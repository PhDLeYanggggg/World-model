# Joint Allocation Helps Accuracy, Not Yet Risk Control

## What Changed

The preceding selection-exchange diagnosis found a benefit loss larger than
the reduction in harm. This experiment uses the magnitude of the already
trained utility estimate to assign the SAME number of interventions in each
locality/recording/current-frame query. Forecasts, fitting sources, risk scores,
eligibility and the nominal 2% all/easy budgets remain frozen. It changes
utility ordering together with within-query risk-slack pooling. It does not
isolate those two allocation mechanisms from each other.

Registration commit: `0cffde4a`. All 108 action groups were frozen in `18801109`
before the first outcome readout. No new training or held-out threshold search.

## Supported Development Result

At identical query-level intervention counts, joint allocation improves ADE
over independent admission by **0.232172% [0.122483%,0.358574%]**. All twelve
locality point contrasts are positive, ranging from 0.000653% to 0.725591%.
It also beats the earlier matched-count risk ranking by 0.2185%
[0.1176%,0.3310%]. These are ratios with different reference denominators,
not simply the difference of separately averaged gains over the floor.

ADE gain over the protected floor increases from 0.2594% to 0.4882%; hard-subset
gain increases from 0.1923% to 0.4996%. Intervention remains 8.0235%. Each of the
three forecasting-seed summaries improves. The mean 95th-percentile error
ratio to the floor decreases from 0.9988 to 0.9963.

The experiment therefore supports utility-aware joint budget allocation as
a development direction. It is not evidence for a large effect, independent
generalization, nonadditive interaction modeling or an original method by itself.

## Why It Is Not Deployable

Actual selected positive-harm budget violations increase from 82 to 99 of 216
dependent views. Ten views still have undefined selected-risk denominators;
the fixed-roster risk summary remains undefined, not zero. Every defined
query-level predicted constraint passes, but predicted feasibility is not
calibration or an observed-risk certificate.

The worst easy gain versus CV remains positive 0.3716%, with no zero-CV harm.
That is net easy preservation, not control of the separate positive-harm
quantity. Unknown-label interventions increase from 23.4954 to 28.0787 per
dependent view on average; those actions remain in coverage counts and their
prediction error is unknown. The scored result cannot establish safety on them.

Unconstrained utility top-k obtains 1.6807% ADE gain over the floor, but 186 views
violate the risk budget and worst easy gain is -3.2797%. Whole-query uniform
admission obtains 0.0429% at 1.1243% intervention, with 59 violations and 46 undefined
views. It is not rate matched. Neither is a deployable alternative.

## Boundary and Next Action

Retain the protected deployment policy. No independent selection, calibration
or confirmation sources were opened. The independent test claim remains
not_run. The historical exposed Stage35/37/43/44 figures are not recertified
by this experiment. Stage5C and SMC remain disabled.

Keep this allocation rule fixed for the next targeted repair: learn and
check query-level harm excess on fitting sources, with source-separated
controller calibration and support-aware abstention. Compare against the
frozen current risk head, retaining count/coverage controls. Register the
source roles and objective before running; do not tune the current readout
or quietly relax the 2% budget. A valid calibration study needs enough distinct
sources and effective queries, not more overlapping windows.

Material passport: fresh allocation and readout, cached_verified estimators.
Twelve opened development localities, three forecaster seeds, 108 paired groups,
3,000 locality-bootstrap draws after dependent-view averaging. Detector silver,
image-local coordinates, obs8/pred12 raw-frame stride12. No metric, seconds,
human-gold, physical-safety, true 3D, foundation or submission-readiness claim.
