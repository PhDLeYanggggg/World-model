# Past-Image Context Under The Original Risk Contract

I tested whether information measured from the past images helps a neural
controller distinguish useful forecasts from harmful interventions. Four
matched arms isolate image bounds, measurement reliability and background
motion. This is an exposed-development experiment, not an independent test.

Registered motion component screen: **FAIL**.
Primary paired completion-lower: **+0.4615pp [-1.0949, +2.2936]**.
Original full motion-policy recording contract: **FAIL**.
**No deployment or submission-readiness change.**

## Complete Four-Arm Decision Matrix

| Inputs | Policy | Known gain % | Separate lower % | Hard gain % | Easy gain % | Switch % | Original pass | Easy harm violations | Easy NA | Worst signed easy loss % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_image | frame_uniform | +0.0066 | +0.0027 | +0.0073 | +0.0021 | +1.0620 | 2/72 | 7 | 57 | +1.0416 |
| no_image | independent | +0.2346 | +0.1877 | +0.1538 | +0.2978 | +19.0175 | 25/72 | 23 | 7 | +8.1828 |
| no_image | joint_free | +0.7004 | +0.5498 | +0.6384 | -0.3085 | +24.5309 | 18/72 | 31 | 3 | +12.4108 |
| no_image | joint_matched | +0.6460 | +0.5059 | +0.6010 | -0.3803 | +19.0175 | 15/72 | 33 | 6 | +11.5679 |
| no_image | proposal_only | +0.0132 | +0.0113 | +0.0112 | +0.0004 | +0.9644 | 21/72 | 12 | 37 | +0.5552 |
| image_bounds | frame_uniform | -0.0004 | -0.0004 | -0.0000 | -0.0236 | +0.0221 | 2/72 | 3 | 67 | +0.7343 |
| image_bounds | independent | +0.3325 | +0.3276 | +0.2622 | +0.3312 | +15.5333 | 36/72 | 13 | 16 | +2.8886 |
| image_bounds | joint_free | +1.2611 | +1.2339 | +2.2284 | -0.6125 | +20.3524 | 26/72 | 24 | 14 | +9.5477 |
| image_bounds | joint_matched | +1.2104 | +1.1864 | +2.1888 | -0.6665 | +15.5333 | 21/72 | 23 | 19 | +9.6490 |
| image_bounds | proposal_only | +0.0172 | +0.0171 | +0.0107 | +0.0060 | +0.7755 | 14/72 | 8 | 47 | +0.3469 |
| quality_geometry | frame_uniform | -0.0001 | -0.0001 | +0.0001 | -0.0136 | +0.0697 | 2/72 | 3 | 67 | +0.7343 |
| quality_geometry | independent | +2.6663 | +2.6088 | +3.2823 | +0.8799 | +24.7712 | 17/72 | 26 | 20 | +25.4081 |
| quality_geometry | joint_free | +5.2899 | +5.1363 | +8.2741 | -1.9276 | +28.8910 | 12/72 | 35 | 19 | +47.3520 |
| quality_geometry | joint_matched | +5.3184 | +5.1654 | +8.2888 | -1.6531 | +24.7712 | 8/72 | 35 | 21 | +38.3389 |
| quality_geometry | proposal_only | +1.3922 | +1.3754 | +1.9682 | +0.0148 | +3.4288 | 11/72 | 23 | 37 | +23.5644 |
| motion_geometry | frame_uniform | +0.0048 | +0.0043 | +0.0055 | -0.0757 | +0.4512 | 2/72 | 8 | 60 | +5.4876 |
| motion_geometry | independent | +3.6949 | +3.6119 | +4.9224 | -1.6845 | +30.7973 | 16/72 | 39 | 10 | +68.0427 |
| motion_geometry | joint_free | +5.8670 | +5.6931 | +9.2883 | -6.5202 | +35.0395 | 12/72 | 43 | 9 | +118.2649 |
| motion_geometry | joint_matched | +5.8245 | +5.6550 | +9.2130 | -6.4116 | +30.7973 | 12/72 | 43 | 12 | +115.7482 |
| motion_geometry | proposal_only | +2.4151 | +2.3653 | +3.6927 | -1.5989 | +5.8059 | 2/72 | 36 | 32 | +61.1388 |

Mean easy utility is not the original per-recording safety check.
The72views repeat24recordings across three risk-head seeds. NA remains
NA. Empty same-count controls cannot establish utility or safety.

## Registered And Secondary Paired Contrasts

| Contrast | Estimand | Point pp | Exploratory source 95% interval |
|---|---|---:|---|
| bounds | completion_lower_pp | +0.3891 | [-0.7152, +1.5393] |
| bounds | completion_upper_pp | +0.7322 | [-0.2025, +1.7402] |
| bounds | known_pp | +0.5607 | [-0.4530, +1.6293] |
| fixed_control | completion_lower_pp | +3.2442 | [-2.9567, +12.4148] |
| fixed_control | completion_upper_pp | +5.6208 | [-0.0385, +14.8167] |
| fixed_control | known_pp | +4.4325 | [-1.3389, +13.5804] |
| full_bundle | completion_lower_pp | +4.8497 | [-1.0236, +14.0402] |
| full_bundle | completion_upper_pp | +5.4834 | [-0.2601, +14.6847] |
| full_bundle | known_pp | +5.1666 | [-0.6255, +14.3814] |
| primary | completion_lower_pp | +0.4615 | [-1.0949, +2.2936] |
| primary | completion_upper_pp | +0.6925 | [-0.8558, +2.6608] |
| primary | known_pp | +0.5770 | [-0.9617, +2.4708] |

Primary: motion_geometry minus quality_geometry under joint_free.
Bounds minus no-image is secondary. Motion minus no-image is a bundle
contrast, not pure motion attribution. The fixed control is the original
guarded1/32control; no coefficient is selected after outcomes.

Completion lower/upper are conservative unknown-label bounds, not estimates
of actual unobserved errors. Intervals use3,000source-component draws after
averaging risk seeds within8groups. They are exploratory, not independent
confirmation or correction for adaptive project-wide model development.

## Primary By Source And Risk Seed

| Source | Known pp | Paired completion-lower pp |
|---|---:|---:|
| moving_source_component_048 | +1.9290 | +1.9281 |
| moving_source_component_053 | +0.1619 | +0.1619 |
| moving_source_component_059 | -0.1520 | -0.1766 |
| moving_source_component_060 | +0.1748 | +0.1359 |
| moving_source_component_065 | +0.1886 | +0.1868 |
| moving_source_component_067 | +6.5888 | +6.1764 |
| moving_source_component_073 | -2.4546 | -2.5808 |
| moving_source_component_076 | -1.8203 | -2.1395 |

| Risk-head seed | Paired completion-lower pp |
|---|---:|
| 17 | +0.0004 |
| 29 | +0.1107 |
| 43 | +1.2735 |

## Actual Training And Reproduction

Fresh_run:24Torch heads,19,814parameters each,4,096updates each(98,304updates),
plus24complete exact refits(another98,304updates). Real recovery128+128vs256
exact:True. The corresponding resumed256-step
pilot prefix is included in its full fit, not counted again as another model.
The separate continuous256-step pilot contributes256additional updates:
196,864actual optimizer updates including recovery and complete refits.
All four arms share capacity, initialization, samplers, labels and budget.
Masked context parameters are inactive; equal capacity does not imply equal
effective information. Normalizers are fitted on A only.

Cached_verified:frozen OOF forecasts, original source/ID schemas and causal
measurement bundles. Fresh measured images were acquired before this fitting.
Not_run:new forecasters, three forecaster seeds, new C/H or reserved outcome
readout, independent confirmation or calibrated physical safety.

Training uses producer-A OOF data only, not own-B labels. Each producer
support excludes its corresponding own-B sources. This differs from the
old50%A/Baugmentation regime; unmatched historical differences are not a
pure image effect. Reference/support heads cannot see candidate-only fields.
All36choice contexts freeze before this complete62,466-row own-B readout.

Cumulative model fitting seconds exclude repeated input loading/preprocessing;
the separate phase receipts report end-to-end process time and peak RSS.

## Interpretation Limits

Native observation8/prediction12, annotation stride1 and image coordinates.
Available-label ADE is not full12ADE or SDD rawt50. No verified meters,
seconds, homography, true3D, foundation, human-gold or collision-safety claim.
Background motion is a conventional image proxy, not calibrated ego-motion.
Original2%all/easy harm, signed easy-net, reference/support and NA rules stay
unchanged. Old failures remain failures. Stage5C execution and SMC are off.

[Protocol](protocol.md) | [Recording checks](recording_checks.csv) |
[Loss trace](loss_trace.csv) | [Failure analysis](failure_analysis.md) |
[Reproduction](operation_zh.md)
