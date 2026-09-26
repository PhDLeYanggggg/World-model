# Support-Fractional Harm: Ranking Lift, Magnitude Gate Failed

## Result and Provenance

The fixed auxiliary does not pass its registered development-advancement gate.
No new policy is evaluated or deployed. This is not a trajectory improvement,
independent confirmation, calibrated safety guarantee or submission-ready result.

Registration `467c10ad` preceded 144 fresh native-Torch fits, each 2,000 updates;
prediction freeze `d9aae4c1` preceded this round's outcome readout. The 144 mean
controls are cached_verified, with identical initialization, sampled rows and
update budget. Six ordered source assignments, three seeds and four held
localities per assignment are retained, including the motion-only comparison.
No favorable role, seed or loss coefficient was selected after readout.

## Primary Endpoint

Positive values below indicate improvement in positive-disagreement easy-harm
MSE over the matched mean head. Each interval resamples four locality means,
after averaging three seeds within locality, 3,000 times. These are overlapping
source-development views, not independent replications or simultaneous CIs.

| Producer / controller | MSE gain (%) | Locality-bootstrap 95% CI (%) |
|---|---:|---:|
| 0 / 1 | -1.605 | [-5.291, 0.646] |
| 0 / 2 | -25.741 | [-51.920, -0.066] |
| 1 / 0 | -0.347 | [-2.649, 2.080] |
| 1 / 2 | -6.081 | [-17.013, 0.789] |
| 2 / 0 | +0.783 | [-0.218, 1.904] |
| 2 / 1 | -20.322 | [-42.179, 1.535] |

There are zero positive, one negative and five overlapping primary intervals.
The requirement was six positive intervals, not a favorable aggregate rank.
Motion-only has one positive and five overlapping primary intervals; it does
not rescue the failed full-pair criterion. Its point range is -50.346% to +2.040%.

## What Improved, and What Did Not

The full head's median conditional event AUROC rises from 0.48578 to 0.61971;
five of six paired AUROC-change intervals are positive, one overlaps zero.
Median conditional AP/prevalence rises from 1.08796 to 1.66631. These are
ranking diagnostics of an expected-harm score, not calibrated probabilities.

Median harm coverage moves from 0.66266 to 0.89098. Absolute log-coverage error
improves with three positive and three overlapping intervals. This does not
establish row-level or selected-set calibration. Conditional top10 harm mass
capture has one positive and five overlapping contrast intervals; one point
is negative. The tail/coverage guard passes only because none of its intervals
is wholly negative or unestimable, not because every role improves.

Reference-cost fitting is not unaffected: D_all MSE has two negative intervals
and no positive interval. The auxiliary shares a network with the original
moments. Do not hide that cost behind the better event ranking.

Training easy-harm MSE improves in 56/72 full views, while held MSE improves
in 33/72. Twenty-nine views improve in fitting but not in the held locality.
These dependent counts support a fitting/transport distinction, not a unique
causal diagnosis. The [fit analysis](fit_transport_diagnosis.md) retains each view.

Of 39 full views with worse held MSE, zero observed easy-harm targets dominate
the excess squared error in 37. Only 5/39 are dominated by the upper causal
envelope partition. A zero easy-harm target can mean a non-easy row or an easy
row with no harm; it does not always mean a harmless row. This distinction
limits the current mechanism claim and guides the next diagnosis.

## Execution and Boundaries

The new fits consumed 417.74 summed fitting seconds, not whole-pipeline wall
time, and completed all 288,000 updates. Unknown-label draws and numerical
target clamps were zero. Native arm64 CPU4, interop1, workers0; no CREATE job.
All frozen readouts and scoped checks are recorded in `verification.json`.

`fresh_run`: new heads, paired readouts, diagnostic decomposition and checks.
`cached_verified`: source data/forecasts and matched controls. `not_run`: new
C policy, deployment, independent confirmation and remote M3W asset inventory.
The remote project path is unverified, not evidence that remote assets do not exist.

The observed ranking improvement motivates separating transferable ranking
from magnitude fitting, not rescaling current held predictions or opening a
policy evaluation despite a failed gate. See [failure analysis](failure_analysis.md)
and [project gap](project_gap.md). Existing deployment is unchanged.

Obs8/pred12 annotation steps, raw stride12, image pixels and detector-derived
labels remain the measurement scope. No metric/seconds, human-gold, physical
safety, true3D or foundation claim. Independent calibration/confirmation stay
closed. Stage5C and SMC remain off.
