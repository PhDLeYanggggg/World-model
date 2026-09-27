# Honest OOF Magnitude Readout

fresh_run:1008 native Torch nuisance/auxiliary heads;2016000 optimizer updates;432 two-slope readouts.
cached_verified:432 cost-only inner controls,144 outer heads per arm,forecasts and causal features.
not_run:new trajectory or policy,independent selection,reserved calibration,confirmation.

| Contrast/family | Positive / negative / overlap / missing CIs | Primary range (%) |
|---|---|---|
| scaled_true_vs_scaled_cost/full | [2, 0, 4, 0] | [-2.9100131617630343, 1.039773918596534] |
| scaled_true_vs_scaled_cost/motion_only | [1, 1, 4, 0] | [-1.4680584681350861, 1.0696372513889976] |
| scaled_true_vs_raw_true/full | [2, 0, 4, 0] | [1.2023693234841997, 27.889954731576587] |
| scaled_true_vs_raw_true/motion_only | [4, 0, 2, 0] | [7.239256811400797, 39.82445169878173] |
| scaled_true_vs_scaled_shuffled/full | [2, 0, 4, 0] | [-3.198485629312045, 1.1250688934602346] |
| scaled_true_vs_scaled_shuffled/motion_only | [1, 2, 3, 0] | [-4.5083127631026905, 1.0106821389802745] |
| scaled_cost_vs_raw_cost/full | [1, 0, 5, 0] | [-1.7738641927727545, 26.21018528529744] |
| scaled_cost_vs_raw_cost/motion_only | [4, 0, 2, 0] | [5.709688204071093, 37.2386875250652] |
| scaled_shuffled_vs_raw_shuffled/full | [2, 0, 4, 0] | [-2.4816289958824687, 27.039728538243608] |
| scaled_shuffled_vs_raw_shuffled/motion_only | [4, 0, 2, 0] | [6.971960579606653, 39.17291854593908] |

## Assignment Intervals

| Contrast/family/assignment | Point (%) | 95% CI |
|---|---:|---|
| scaled_true_vs_scaled_cost/full/producer0_controller1 | 0.12842557958188436 | [0.05498549264668352, 0.2504512745680452] |
| scaled_true_vs_scaled_cost/full/producer0_controller2 | -0.016764054244541342 | [-0.05580240987136449, 0.004155471294555037] |
| scaled_true_vs_scaled_cost/full/producer1_controller0 | 1.039773918596534 | [0.07525777815644873, 2.7367567421100256] |
| scaled_true_vs_scaled_cost/full/producer1_controller2 | -0.022634064038052792 | [-0.10684397861789582, 0.028818919024099048] |
| scaled_true_vs_scaled_cost/full/producer2_controller0 | -1.931539425022907 | [-7.37413067415031, 1.2335008349509131] |
| scaled_true_vs_scaled_cost/full/producer2_controller1 | -2.9100131617630343 | [-6.82525974491095, 1.0052334213848817] |
| scaled_true_vs_scaled_cost/motion_only/producer0_controller1 | -1.4680584681350861 | [-4.336418122604118, 0.0013782356654574186] |
| scaled_true_vs_scaled_cost/motion_only/producer0_controller2 | 0.03191456454127807 | [0.004513083690615071, 0.07388778207145484] |
| scaled_true_vs_scaled_cost/motion_only/producer1_controller0 | 1.0696372513889976 | [-0.14236450043496485, 3.326968979627093] |
| scaled_true_vs_scaled_cost/motion_only/producer1_controller2 | -0.22899012317963668 | [-0.6913686884265537, 0.00473805706010473] |
| scaled_true_vs_scaled_cost/motion_only/producer2_controller0 | 0.13040061450655518 | [-0.021732132566738883, 0.41345146874007055] |
| scaled_true_vs_scaled_cost/motion_only/producer2_controller1 | -0.3856766783488186 | [-1.0944141399653153, -0.002467302308685365] |
| scaled_true_vs_raw_true/full/producer0_controller1 | 1.2023693234841997 | [0.001233568503758449, 3.0751826431539566] |
| scaled_true_vs_raw_true/full/producer0_controller2 | 27.889954731576587 | [0.07515638613637411, 57.23420119791815] |
| scaled_true_vs_raw_true/full/producer1_controller0 | 7.143751400640257 | [-0.7210422400461155, 15.008545041326633] |
| scaled_true_vs_raw_true/full/producer1_controller2 | 3.474715989032153 | [-0.16752559544175505, 8.645811493472152] |
| scaled_true_vs_raw_true/full/producer2_controller0 | 4.47310321023714 | [-7.0315429906359554, 15.201523224093112] |
| scaled_true_vs_raw_true/full/producer2_controller1 | 4.613575004142663 | [-4.772651772234605, 14.383712338418071] |
| scaled_true_vs_raw_true/motion_only/producer0_controller1 | 27.144814293826 | [0.18557672078323992, 71.18316836565624] |
| scaled_true_vs_raw_true/motion_only/producer0_controller2 | 39.82445169878173 | [0.7654311032558686, 78.88347229430758] |
| scaled_true_vs_raw_true/motion_only/producer1_controller0 | 13.051180953420733 | [-0.6157832140252587, 26.92783500683626] |
| scaled_true_vs_raw_true/motion_only/producer1_controller2 | 7.239256811400797 | [-0.005421430690460315, 20.852650075798927] |
| scaled_true_vs_raw_true/motion_only/producer2_controller0 | 17.553526574020406 | [1.76159835738865, 36.38545916679827] |
| scaled_true_vs_raw_true/motion_only/producer2_controller1 | 26.248896828710528 | [0.03004725589909918, 52.467746401521964] |
| scaled_true_vs_scaled_shuffled/full/producer0_controller1 | 0.22344030637246554 | [0.07489944632859677, 0.37198116641633433] |
| scaled_true_vs_scaled_shuffled/full/producer0_controller2 | -0.00023599484195596307 | [-0.06766966645255741, 0.06578658003346521] |
| scaled_true_vs_scaled_shuffled/full/producer1_controller0 | 1.1250688934602346 | [0.11150120583338657, 2.2510470137637197] |
| scaled_true_vs_scaled_shuffled/full/producer1_controller2 | -0.01126015480638401 | [-0.07893439168810713, 0.03044257883180234] |
| scaled_true_vs_scaled_shuffled/full/producer2_controller0 | -0.6590392305262583 | [-4.910345211232142, 2.608933437990788] |
| scaled_true_vs_scaled_shuffled/full/producer2_controller1 | -3.198485629312045 | [-6.9063878512002415, 0.5094165925761515] |
| scaled_true_vs_scaled_shuffled/motion_only/producer0_controller1 | -4.5083127631026905 | [-13.397610971794226, -0.01237463175209508] |
| scaled_true_vs_scaled_shuffled/motion_only/producer0_controller2 | 0.0049558878898928755 | [-0.0012590001830166907, 0.014010182012850798] |
| scaled_true_vs_scaled_shuffled/motion_only/producer1_controller0 | 1.0106821389802745 | [0.03364110850828187, 2.740840538303964] |
| scaled_true_vs_scaled_shuffled/motion_only/producer1_controller2 | -0.27716828080957306 | [-0.8362567617735042, 0.003941990829280696] |
| scaled_true_vs_scaled_shuffled/motion_only/producer2_controller0 | 0.4926193690480704 | [-0.007993922621298942, 1.4886388686787617] |
| scaled_true_vs_scaled_shuffled/motion_only/producer2_controller1 | -0.411830542369088 | [-1.1671369386558847, -0.016510748275204625] |
| scaled_cost_vs_raw_cost/full/producer0_controller1 | 0.9089444114048718 | [-0.0030774411500411394, 2.2866530200323774] |
| scaled_cost_vs_raw_cost/full/producer0_controller2 | 26.21018528529744 | [0.04706752797301374, 56.7093790076819] |
| scaled_cost_vs_raw_cost/full/producer1_controller0 | 6.998952303486185 | [-0.7763494681797398, 14.774254075152111] |
| scaled_cost_vs_raw_cost/full/producer1_controller2 | 3.2989510616984887 | [-0.13555028311256873, 7.807597380438724] |
| scaled_cost_vs_raw_cost/full/producer2_controller0 | -1.7738641927727545 | [-7.482965108818874, 2.849351389041305] |
| scaled_cost_vs_raw_cost/full/producer2_controller1 | -0.2947831098005338 | [-9.015292953021426, 8.425726733420358] |
| scaled_cost_vs_raw_cost/motion_only/producer0_controller1 | 26.88736167903469 | [0.12703980237565488, 71.0874057968966] |
| scaled_cost_vs_raw_cost/motion_only/producer0_controller2 | 37.2386875250652 | [0.24296063592283812, 74.23441441420755] |
| scaled_cost_vs_raw_cost/motion_only/producer1_controller0 | 12.952261388522823 | [-0.4477017940127512, 27.93143736675551] |
| scaled_cost_vs_raw_cost/motion_only/producer1_controller2 | 5.709688204071093 | [-0.00557025876273931, 16.31093108186736] |
| scaled_cost_vs_raw_cost/motion_only/producer2_controller0 | 16.043847784970968 | [1.553769305302817, 33.21222585700174] |
| scaled_cost_vs_raw_cost/motion_only/producer2_controller1 | 23.95455530442796 | [0.08852274858400072, 52.00075507371599] |
| scaled_shuffled_vs_raw_shuffled/full/producer0_controller1 | 1.0068877170689423 | [0.04831305002686238, 2.532056915919316] |
| scaled_shuffled_vs_raw_shuffled/full/producer0_controller2 | 27.039728538243608 | [0.2650691661316934, 59.533330125893805] |
| scaled_shuffled_vs_raw_shuffled/full/producer1_controller0 | 6.088704523101326 | [-1.0158610292873986, 13.19327007549005] |
| scaled_shuffled_vs_raw_shuffled/full/producer1_controller2 | 2.82889380764017 | [-0.11650799657980365, 6.6857330550318155] |
| scaled_shuffled_vs_raw_shuffled/full/producer2_controller0 | -2.4816289958824687 | [-8.579407471537671, 1.873902357691421] |
| scaled_shuffled_vs_raw_shuffled/full/producer2_controller1 | 0.3703458301156659 | [-8.600150569351952, 9.340842229583284] |
| scaled_shuffled_vs_raw_shuffled/motion_only/producer0_controller1 | 26.95235468748676 | [0.0697861228748817, 72.19149985924881] |
| scaled_shuffled_vs_raw_shuffled/motion_only/producer0_controller2 | 39.17291854593908 | [0.2557087380748459, 78.09012835380332] |
| scaled_shuffled_vs_raw_shuffled/motion_only/producer1_controller0 | 13.633033663231434 | [-0.38647573958972775, 27.652543066052594] |
| scaled_shuffled_vs_raw_shuffled/motion_only/producer1_controller2 | 6.971960579606653 | [-0.0004984419348961679, 19.713249732528777] |
| scaled_shuffled_vs_raw_shuffled/motion_only/producer2_controller0 | 16.752670583849234 | [1.5555560713587218, 36.30676292756764] |
| scaled_shuffled_vs_raw_shuffled/motion_only/producer2_controller1 | 23.222278756740028 | [0.08944581690985823, 51.3927641908988] |

## Boundaries
Primary is positive-envelope expected easy-harm cost MSE,not FDE/ADE or easy degradation.
Three seeds averaged within locality;3000 paired four-locality resamples per assignment.
Assignments overlap;source-held rows were exposed historically. All CIs are descriptive and unadjusted.
Single-site reference to two-site OOF head to three-site outer head introduces training-size and easy-cut drift.
No labels from inner/outer held localities entered their producer lineage. OOF targets use producer-specific easy cuts.
Origin least squares is followed by nested causal-envelope clipping;it is not a conformal or safety guarantee.
Obs8/pred12 native steps,detector pixels. No metric/seconds,true3D,foundation or human-gold claims.

```json
{
  "primary_cost_gate": false,
  "guard_gate": false,
  "information_gate": false,
  "advance_gate": false,
  "deployment_changed": false,
  "independent_confirmation": false,
  "submission_ready": false,
  "stage5c_executed": false,
  "smc_enabled": false
}
```
