# Source Decoder by Risk Objective: Complete Fixed-Final Readout

## Execution and Provenance

Fresh_run:72 cost-head fits,144,000 updates in 539.88s;144 fit/held readouts in 111.02s. All72 finals were frozen before inference.
The18 shared-risk models reproduce their old controls exactly: weights, optimizer,
duals, traces and samplers. Cached_verified source-local heads are unchanged.
Across all four arms, source queries, preprocessing and initial predictions match.
Four source decoders add495 parameters; this is not a parameter-matched test.
No new trajectory producer, JEPA representation, residual or latent rollout ran.

The head predicts benefit B, positive harm H, floor-reference R, easy
reference R_easy and easy harm H_easy. Signed scores are B-H, H-0.02R
and H_easy-0.02R_easy. Reported score MSE is not trajectory FDE/ADE.

Original source TRAIN is partitioned by whole recording. Both fit and held
recordings remain TRAIN-internal; held data have prior research exposure.
No outer development or independent selection/calibration/confirmation was read.
Preprocessing uses inner-fit only. Unknown outcomes remain in inference and
completion bounds; future supervision is not an inference feature.

## Fit Recording Results

897,060 repeated row occurrences/arm, including 18,978 unknown outcomes. Occurrences overlap across context/head-seed views.

| Contrast | Full lower utility, pp [95% CI] | Same-count lower utility, pp [CI] | Signed-score MSE difference [CI] |
|---|---:|---:|---:|
| source_risk minus shared_risk | -1.209068 [-2.291125, -0.090422] | -0.364152 [-0.768320, +0.001407] | +1.168036 [-0.707380, +4.576373] |
| source_utility minus shared_utility | -2.318915 [-5.750776, +0.194764] | -0.268728 [-0.924475, +0.424329] | -0.073962 [-0.107675, -0.044951] |
| shared_risk minus shared_utility | -3.399677 [-4.773970, -2.054756] | -0.559569 [-1.131721, -0.068255] | +1.476325 [+0.839168, +2.458395] |
| source_risk minus source_utility | -6.076564 [-9.826572, -3.087018] | -1.232757 [-2.386142, -0.367156] | +2.718323 [+0.507084, +6.939694] |
| source_risk minus local | -3.939537 [-6.055402, -2.270951] | -1.460783 [-2.613931, -0.490360] | +2.573931 [+0.324882, +6.851444] |
| source_utility minus local | -3.384274 [-7.178149, -0.495246] | -0.926560 [-2.024176, -0.099301] | -0.144392 [-0.267272, -0.025461] |

Decoder-by-risk MSE interaction: +1.241998 [-0.619466, +4.633595].

| Arm | Selected | Known easy violations /72 | Upper easy violations /72 | Undefined easy /72 | Full support /72 | Mean view MSE |
|---|---:|---:|---:|---:|---:|---:|
| local | 123490 | 1 | 59 | 7 | 5 | 0.559739 |
| shared_risk | 67774 | 9 | 57 | 7 | 7 | 1.965634 |
| shared_utility | 337984 | 62 | 65 | 6 | 1 | 0.489310 |
| source_risk | 15364 | 1 | 54 | 12 | 6 | 3.133671 |
| source_utility | 176250 | 60 | 67 | 5 | 0 | 0.415347 |
## Held Recording Results

419,766 repeated row occurrences/arm, including 9,228 unknown outcomes. Occurrences overlap across context/head-seed views.

| Contrast | Full lower utility, pp [95% CI] | Same-count lower utility, pp [CI] | Signed-score MSE difference [CI] |
|---|---:|---:|---:|
| source_risk minus shared_risk | -0.696236 [-1.704915, +0.576560] | -0.176495 [-0.436163, +0.111590] | +5.709067 [-1.488146, +15.219810] |
| source_utility minus shared_utility | -1.762683 [-2.519183, -0.977941] | -0.580246 [-1.100566, -0.057594] | +0.256604 [-0.096676, +0.883664] |
| shared_risk minus shared_utility | -3.006812 [-3.952084, -2.054663] | -0.505483 [-0.964313, -0.075610] | +5.756850 [+3.125110, +8.602720] |
| source_risk minus source_utility | -3.096195 [-5.235898, -1.409567] | -0.700032 [-1.234201, -0.236819] | +11.209313 [+3.046555, +21.855650] |
| source_risk minus local | -2.114539 [-3.272387, -1.011576] | -0.526203 [-0.821241, -0.240731] | +9.147241 [+1.054802, +20.153037] |
| source_utility minus local | -1.287145 [-3.052476, +0.123288] | -0.568936 [-1.078989, -0.109548] | -2.062072 [-3.349375, -0.956808] |

Decoder-by-risk MSE interaction: +5.452463 [-1.700635, +15.012943].

| Arm | Selected | Known easy violations /72 | Upper easy violations /72 | Undefined easy /72 | Full support /72 | Mean view MSE |
|---|---:|---:|---:|---:|---:|---:|
| local | 57903 | 46 | 65 | 7 | 0 | 4.013844 |
| shared_risk | 27826 | 40 | 56 | 12 | 1 | 7.452018 |
| shared_utility | 154772 | 61 | 71 | 1 | 0 | 1.695168 |
| source_risk | 17643 | 42 | 59 | 12 | 1 | 13.161085 |
| source_utility | 82801 | 54 | 68 | 1 | 3 | 1.951772 |

## Interpretation Boundary

Utility is a pair-specific completion-lower difference divided by full known
floor-reference error mass, not ADE/FDE improvement. Shared unknown actions
cancel in a comparison; unmatched unknown actions retain worst-case bounds.
Pair-specific lower utilities are not transitive and must not be subtracted
to create an interaction. Only the MSE difference-in-differences is reported.
The nominal3,000-draw bootstrap averages repeated views within12 localities.
Shared training, reused held recordings and multiple contrasts limit inference:
these are exposed TRAIN-internal intervals, not independent significance.
Three cost-head seeds are not three independent forecasting-model seeds.

The2% positive-harm/reference evaluation budget is identical in every arm.
Removing a training penalty does not relax safety. Undefined/empty risk is
not a pass. Full support additionally requires easy preservation and positive
completion utility; same-count risk is diagnostic, not a second deployment gate.

Final sampled training losses are retained in verified_details.json.
The risk objective includes a signed Lagrangian penalty and can be negative;
its total loss is not directly comparable with the utility-only total.
Last-batch loss is not a full-training-census error. Easy-harm calibration
ratios also use different selected cohorts in different arms, not a controlled
fixed-cohort calibration comparison.

## Registered Routing Decision

- source_risk: eligible for later development design = False.
  Reasons: fit_decoder_prediction_lift_not_supported, held_decoder_prediction_lift_not_supported, full_utility_vs_shared_risk_not_supported, full_utility_vs_local_not_supported, matched_utility_vs_shared_risk_not_supported, matched_utility_vs_local_not_supported, full_policy_risk_support_failed.
- source_utility: eligible for later development design = False.
  Reasons: held_decoder_prediction_lift_not_supported, full_utility_vs_shared_utility_not_supported, full_utility_vs_local_not_supported, matched_utility_vs_shared_utility_not_supported, matched_utility_vs_local_not_supported, full_policy_risk_support_failed.

No deployment changed. A routing pass would still not authorize an
independent-test claim or establish M3W/CVPR success.

## Held Source Slices

| Locality | Local | Shared risk | Shared utility | Source risk | Source utility |
|---|---:|---:|---:|---:|---:|
| eu-locality-007 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| eu-locality-008 | 0/6 | 1/6 | 6/6 | 1/6 | 0/6 |
| eu-locality-020 | 0/6 | 0/6 | 0/6 | 0/6 | 1/6 |
| eu-locality-048 | 5/6 | 6/6 | 6/6 | 3/6 | 6/6 |
| eu-locality-067 | 6/6 | 1/6 | 6/6 | 6/6 | 6/6 |
| eu-locality-074 | 0/6 | 5/6 | 6/6 | 1/6 | 6/6 |
| eu-locality-082 | 3/6 | 0/6 | 2/6 | 0/6 | 2/6 |
| eu-locality-110 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| eu-locality-112 | 5/6 | 0/6 | 6/6 | 4/6 | 6/6 |
| eu-locality-119 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| eu-locality-124 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| eu-locality-126 | 3/6 | 3/6 | 5/6 | 3/6 | 3/6 |

Cells show known easy-risk violations. The JSON retains undefined risk,
completion support, unknown selections and calibration diagnostics separately.

## Reproduce and Inspect

```bash
.venv-pytorch/bin/python -m scripts.evaluate_m3w_decoder_risk_factorial check
.venv-pytorch/bin/python -m scripts.report_m3w_decoder_risk_factorial --check
```

Completed artifacts are immutable; an interrupted unfinished phase uses its
explicit --resume option. Checkpoints remain local and excluded from Git.
This verifier runs no optimizer and opens no new data role.

Observation8/prediction12, raw-frame stride12, dataset-local coordinates.
Historical t+50 remains separate. No metric/seconds, human-gold, true3D,
foundation or physical-safety claim. Stage5C and SMC remain disabled.
