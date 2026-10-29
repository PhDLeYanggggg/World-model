# Protected Harm Continuation: Complete Readout

## Execution and Evidence

Fresh_run:54 continuations/108,000 new updates in526.97s;144 fit/held views in118.26s after all finals were frozen.
A float32-only precursor failed after37 finals; none is reused here. Its
backward arithmetic was repaired with float64 ratio intermediates, then all54
fits restarted. Weights and outputs remain float32; the failed batch is preserved.
All three arms start from the same shared-utility anchors and continue identical
source/query draws with fresh optimizers. Each continued head has4,000 effective
updates including its2,000-update anchor. Local, old-risk and anchor controls
remain cached_verified2,000-update controls and reproduce their old inference.
Utility continuation and joint risk are the equal-additional-budget controls.
Total parameters25,407 in all arms; isolated harm trains12,258 versus24,615
in the joint/utility arms. This is not an equal-trainable-capacity comparison.

Benefit B, positive harm H, floor error R, easy floor R_easy and easy harm
H_easy are cost moments. Signed scores are B-H, H-0.02R and H_easy-0.02R_easy.
Isolated B/R/R_easy remain bit-exact to the anchor on every evaluated row.
H changes, so net gain is not frozen. No trajectory predictor was trained.

Original TRAIN is split by whole recording; held recordings were previously
exposed. This is exploratory mechanism evidence, not independent confirmation.
Distinct source/recording pairs: 74 fit and 31 held, across12 source localities.
No outer development, independent selection/calibration/test or deployment
changed. Unknown labels remain eligible at inference and receive bounds.

## Fit Recording Results

897,060 repeated row occurrences/arm;18,978 unknown.

| Arm | Mean signed MSE | Selected | Known easy violations /72 | Upper easy violations /72 | Undefined easy /72 | Full support /72 |
|---|---:|---:|---:|---:|---:|---:|
| local | 0.559739 | 123490 | 1 | 59 | 7 | 5 |
| anchor | 0.489310 | 337984 | 62 | 65 | 6 | 1 |
| old_risk | 1.965634 | 67774 | 9 | 57 | 7 | 7 |
| utility_continue | 0.396041 | 401255 | 65 | 66 | 6 | 0 |
| joint_risk | 1.224959 | 76621 | 6 | 62 | 6 | 4 |
| isolated_risk | 1.985264 | 45395 | 21 | 54 | 6 | 11 |

| Contrast | Full lower utility pp [95% CI] | Same-count lower utility pp [CI] | Signed MSE difference [CI] |
|---|---:|---:|---:|
| isolated_risk minus anchor | -4.289350 [-5.294506, -3.226509] | -0.718254 [-1.106582, -0.396032] | +1.495954 [+0.897080, +2.185813] |
| isolated_risk minus utility_continue | -6.820654 [-8.388039, -5.304713] | -1.181593 [-1.842296, -0.623846] | +1.589222 [+0.995465, +2.272818] |
| isolated_risk minus joint_risk | -2.456263 [-3.429208, -1.508309] | -0.381336 [-0.621091, -0.177668] | +0.760305 [+0.189588, +1.440591] |
| isolated_risk minus local | -5.310511 [-7.421131, -3.178151] | -0.962052 [-1.742435, -0.325833] | +1.425524 [+0.815939, +2.132130] |
| isolated_risk minus old_risk | -1.675442 [-2.560730, -0.785560] | -0.147066 [-0.384160, +0.089181] | +0.019630 [-0.933685, +0.833148] |
| joint_risk minus utility_continue | -5.120485 [-6.635549, -3.634600] | -0.857316 [-1.536377, -0.310057] | +0.828917 [+0.490000, +1.317239] |
| utility_continue minus anchor | +1.323332 [+0.380716, +2.514710] | +0.580073 [+0.090100, +1.308960] | -0.093268 [-0.121427, -0.070916] |

### Same Anchor-Selected Cohort: Harm Diagnosis

| Arm | Median predicted / observed easy harm | Defined views /72 | Known selected occurrences |
|---|---:|---:|---:|
| local | 1.680920 | 66 | 332621 |
| anchor | 0.010185 | 66 | 332621 |
| old_risk | 3.772157 | 66 | 332621 |
| utility_continue | 0.003508 | 66 | 332621 |
| joint_risk | 2.260718 | 66 | 332621 |
| isolated_risk | 1.206643 | 66 | 332621 |

These ratios use exactly the same anchor-selected cohort within each view.
They are aggregate harm diagnostics, not ECE or a population calibration guarantee.
Own-selected ratios are retained separately in verified_details.json.

## Held Recording Results

419,766 repeated row occurrences/arm;9,228 unknown.

| Arm | Mean signed MSE | Selected | Known easy violations /72 | Upper easy violations /72 | Undefined easy /72 | Full support /72 |
|---|---:|---:|---:|---:|---:|---:|
| local | 4.013844 | 57903 | 46 | 65 | 7 | 0 |
| anchor | 1.695168 | 154772 | 61 | 71 | 1 | 0 |
| old_risk | 7.452018 | 27826 | 40 | 56 | 12 | 1 |
| utility_continue | 1.967093 | 182163 | 61 | 70 | 2 | 0 |
| joint_risk | 4.663019 | 36085 | 45 | 60 | 9 | 0 |
| isolated_risk | 5.047722 | 24204 | 47 | 63 | 4 | 3 |

| Contrast | Full lower utility pp [95% CI] | Same-count lower utility pp [CI] | Signed MSE difference [CI] |
|---|---:|---:|---:|
| isolated_risk minus anchor | -3.196865 [-3.959617, -2.359196] | -0.593016 [-0.864727, -0.316044] | +3.352554 [+1.871711, +5.396748] |
| isolated_risk minus utility_continue | -4.396890 [-5.709045, -3.001224] | -0.759962 [-1.087357, -0.417726] | +3.080630 [+1.730326, +4.900671] |
| isolated_risk minus joint_risk | -1.627860 [-2.463811, -0.853076] | -0.276264 [-0.490781, -0.076560] | +0.384704 [-0.377030, +1.133890] |
| isolated_risk minus local | -2.833917 [-4.435894, -1.405206] | -0.435754 [-0.901031, -0.097294] | +1.033878 [-1.019736, +3.462073] |
| isolated_risk minus old_risk | -0.912132 [-1.448588, -0.415093] | -0.037968 [-0.139970, +0.073410] | -2.404296 [-5.150958, -0.087748] |
| joint_risk minus utility_continue | -3.368490 [-4.396222, -2.312598] | -0.494007 [-0.855631, -0.177012] | +2.695926 [+1.380347, +4.261844] |
| utility_continue minus anchor | +0.026941 [-0.819315, +0.763733] | +0.271959 [-0.085601, +0.754654] | +0.271924 [+0.056614, +0.548500] |

### Same Anchor-Selected Cohort: Harm Diagnosis

| Arm | Median predicted / observed easy harm | Defined views /72 | Known selected occurrences |
|---|---:|---:|---:|
| local | 1.956680 | 68 | 152061 |
| anchor | 0.004896 | 68 | 152061 |
| old_risk | 4.992457 | 68 | 152061 |
| utility_continue | 0.002744 | 68 | 152061 |
| joint_risk | 2.440403 | 68 | 152061 |
| isolated_risk | 1.530911 | 68 | 152061 |

These ratios use exactly the same anchor-selected cohort within each view.
They are aggregate harm diagnostics, not ECE or a population calibration guarantee.
Own-selected ratios are retained separately in verified_details.json.

## Registered Decision

Eligible for later development design: **False**.
Reasons: full_utility_vs_anchor_not_supported, matched_utility_vs_anchor_not_supported, full_utility_vs_utility_continue_not_supported, matched_utility_vs_utility_continue_not_supported, full_utility_vs_joint_risk_not_supported, matched_utility_vs_joint_risk_not_supported, full_utility_vs_local_not_supported, matched_utility_vs_local_not_supported, full_utility_vs_old_risk_not_supported, matched_utility_vs_old_risk_not_supported, full_policy_risk_support_failed.

Every full/same-count comparison against all five controls must improve, with
full-policy support in all72 held views. MSE improvements alone cannot pass.
The2% positive-harm/reference budget is unchanged; empty or undefined risk is
not safety. No source, seed, zero-action or missing-label case was discarded.

Utility is the pair-specific completion-lower difference divided by full-known
floor reference mass, NOT ADE/FDE improvement. Pair-specific lower bounds are
not transitive. Bootstrap3,000 averages repeated views within12 localities.
Intervals are nominal and exposed TRAIN-internal, not simultaneous/independent.
Three cost-head seeds do not imply three newly trained forecasting producers.

Final sampled losses are in verified_details.json. Signed Lagrangian totals
may be negative and are not comparable objective scales across risk/utility
arms; last-batch loss is not full-census prediction error.

## Held Locality Easy-Risk Failures

| Locality | Local | Anchor | Old risk | Utility continued | Joint risk | Isolated risk |
|---|---:|---:|---:|---:|---:|---:|
| eu-locality-007 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| eu-locality-008 | 0/6 | 6/6 | 1/6 | 6/6 | 2/6 | 3/6 |
| eu-locality-020 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| eu-locality-048 | 5/6 | 6/6 | 6/6 | 6/6 | 5/6 | 2/6 |
| eu-locality-067 | 6/6 | 6/6 | 1/6 | 6/6 | 4/6 | 5/6 |
| eu-locality-074 | 0/6 | 6/6 | 5/6 | 6/6 | 5/6 | 5/6 |
| eu-locality-082 | 3/6 | 2/6 | 0/6 | 1/6 | 0/6 | 0/6 |
| eu-locality-110 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| eu-locality-112 | 5/6 | 6/6 | 0/6 | 6/6 | 1/6 | 6/6 |
| eu-locality-119 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 4/6 |
| eu-locality-124 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| eu-locality-126 | 3/6 | 5/6 | 3/6 | 6/6 | 4/6 | 4/6 |

## Reproduction

```bash
.venv-pytorch/bin/python -m scripts.evaluate_m3w_isolated_harm_fp64 check
.venv-pytorch/bin/python -m scripts.report_m3w_isolated_harm_fp64 --check
```

Completed artifacts are immutable; interrupted incomplete runs use --resume.
Verification loads frozen checkpoints but performs no fit or fresh inference.
Native arm64 CPU4/interop1/workers0; checkpoints excluded from Git.

Observation8/prediction12, raw-frame stride12, dataset-local coordinates.
No metric/seconds, physical-safety, human-gold, true3D or foundation claim.
Stage5C and SMC disabled. Scientific/project success remains unproven.
