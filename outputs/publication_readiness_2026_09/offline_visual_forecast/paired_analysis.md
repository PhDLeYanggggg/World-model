# Paired Visual Contribution and Uncertainty

Equal physical-scene means, then equal training-seed means. 2,000 paired scene resamples.
Only three historical fit scenes: exploratory intervals, not a deployment certificate.

| Contrast | Gain (%) | Scene-bootstrap 95% interval (%) | Seed gains (%) |
| --- | ---: | --- | --- |
| mask_only_vs_geometry | 0.0075 | [-0.006502055589874267, 0.1267401018033243] | [0.019810164282507703, 0.005393931027719567, -0.002728292646825281] |
| current_rgb_vs_geometry | -0.0855 | [-0.2608448581389755, 1.0560667558805914] | [-0.08584638789916887, -0.03521453115413742, -0.135559347307046] |
| past_rgb_vs_geometry | -0.1625 | [-0.26573225462240835, 0.41938916787250813] | [-0.16021081518895475, -0.09510955868001769, -0.2321757512779854] |
| current_rgb_vs_mask_only | -0.0930 | [-0.2543262661139023, 0.9305059783014547] | [-0.10567748706549196, -0.04061065269245212, -0.132827430739213] |
| past_rgb_vs_mask_only | -0.1700 | [-0.2592133448367395, 0.29302044047374576] | [-0.18005664898943508, -0.1005089110890589, -0.22944119880381475] |
| past_rgb_vs_current_rgb | -0.0769 | [-0.6434730934308464, -0.004874681128352876] | [-0.07430064287168303, -0.05987394319748596, -0.09648560870951783] |

The denominator is the named neural reference, not CV. See the main report for absolute baseline comparisons.
A less damaging visual model is not automatically better than a strong causal baseline.
