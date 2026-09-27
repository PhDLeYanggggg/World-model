# Direct Risk-Budget Objective: Matched Results

Fresh_run:144 risk-score heads with288000 updates and comparative scoring.
Cached_verified: the two-moment control weights and frozen forecasting bank; all controls
have fresh inference verified against their old predictions. No new forecaster was fitted.

## Primary Mechanistic Contrast

Signed score is positive harm minus0.02CV ADE. Positive MSE improvement means lower error
than the matched two-moment controller. Roles, preprocessing, sampling, initialization,
architecture and update budget are unchanged. Only the objective differs. Each source
averages dependent producer/seed views before the3000-draw12-locality bootstrap.

| Candidate | Fitting MSE improvement % | Held MSE improvement % |
|---|---:|---:|
| dimensionless | 27.2501 [22.8295, 31.4182] | -3.0567 [-12.3214, 3.9083] |
| damped | 32.4815 [27.8443, 37.7556] | 7.1830 [2.6455, 12.2670] |

## Fixed All-Risk Screen

Both objectives use score<=0. These all-risk-only screens omit full utility/easy guards
and are not proposed deployable selectors. Positive-harm/reference is not net ADE degradation.

| Candidate | Objective | Held coverage % | Held screened positive harm % | Held all ADE gain vs CV % | Held easy gain % | Held hard gain % |
|---|---|---:|---:|---:|---:|---:|
| dimensionless | two_moment_MSE | 21.9765 [17.7182, 26.7335] | 4.8669 [3.1898, 6.9156] | 0.8591 [0.6222, 1.0881] | 4.3498 [1.9270, 6.5856] | 0.9476 [0.6065, 1.2860] |
| dimensionless | signed_excess_MSE | 24.9790 [19.5916, 31.1110] | 3.5227 [2.4851, 4.6424] | 1.0896 [0.6276, 1.6749] | 5.8866 [4.1367, 7.7370] | 0.9104 [0.3235, 1.7543] |
| damped | two_moment_MSE | 55.0798 [49.9696, 60.8483] | 2.6024 [0.9672, 5.3789] | 2.2262 [0.7832, 3.2343] | 3.3811 [0.9933, 4.9438] | 2.4567 [1.4043, 3.3456] |
| damped | signed_excess_MSE | 68.6086 [63.5104, 73.3340] | 1.9357 [1.0733, 3.2058] | 4.6041 [3.9290, 5.1373] | 4.2809 [2.9153, 5.3558] | 4.7749 [4.0469, 5.3673] |

## Source Fitting Versus Holdout

| Candidate | Objective | Fitting positive harm % | Held positive harm % |
|---|---|---:|---:|
| dimensionless | control | 2.8503 [2.3832, 3.2920] | 4.8669 [3.1898, 6.9156] |
| dimensionless | new | 2.8498 [2.1111, 3.5537] | 3.5227 [2.4851, 4.6424] |
| damped | control | 1.0246 [0.9137, 1.1561] | 2.6024 [0.9672, 5.3789] |
| damped | new | 1.0505 [0.8716, 1.2592] | 1.9357 [1.0733, 3.2058] |

## Three Seeds

| Candidate | Seed | Held signed-MSE improvement % |
|---|---:|---:|
| dimensionless | 17 | 0.1475 [-3.5192, 3.7664] |
| dimensionless | 29 | -5.4158 [-20.0798, 4.5676] |
| dimensionless | 43 | -3.9017 [-14.4583, 3.9785] |
| damped | 17 | 7.4380 [2.0739, 13.4093] |
| damped | 29 | 6.1761 [1.6378, 11.5048] |
| damped | 43 | 7.9350 [3.2820, 12.7298] |

## Individual-View Failures

The locality-mean easy gate averages dependent producer/seed views; it does not certify every view.

| Candidate | Worst view easy gain % | Views with easy degradation >2% | Zero-reference harmed row-views |
|---|---:|---:|---:|
| dimensionless | -5.6144 | 5/72 | 1 |
| damped | -3.0685 | 3/72 | 0 |

## Screens and Limitations

- primary_signed_score_MSE_improved: False
- every_locality_nonzero_coverage: True
- every_locality_screen_positive_harm_within_2percent: False
- every_locality_easy_within_2percent: True
- no_zero_reference_harm: False
- exploratory_loss_and_screen_checks_pass: False
- calibration_certificate: False
- full_policy_tested: False
- independent_confirmation: False
- deployment_changed: False
- stage5c_executed: False
- smc_enabled: False

The new two components are an internal parameterization: only their signed combination
is supervised, so interpreting either component as a calibrated moment is invalid.
A loss improvement does not establish reliable accepted-subset risk, independent calibration,
a neural full-policy advantage, or scene-joint benefit. No held outcome selected a threshold
or checkpoint. Unknown labels remain excluded; zero-reference harm stays explicit.
Outer readout and independent roles remain unused. Silver image-local obs8/pred12 at
raw-frame stride12, not metric, seconds, human gold, physical safety, true3D or foundation.
No deployment change, Stage5C execution or SMC.
