# Strong-Base Cap Auxiliary Results

## Material Passport
fresh_run:432 native-Torch heads/864000 updates, original-control reconstruction and144 source-held readouts.
cached_verified:source forecasts, original controls and row-locality-excluded event producers.
not_run:new forecasting, policy evaluation, independent selection/calibration/confirmation.

| Contrast / family | Primary positive / negative / overlap / missing | Point range (%) |
|---|---|---|
| aux_vs_control/full | [1, 1, 4, 0] | [-14.853824484770112, 0.6485206028038051] |
| aux_vs_control/motion_only | [0, 1, 5, 0] | [-21.465514378724137, 2.548059054463529] |
| aux_vs_original/full | [1, 1, 4, 0] | [-14.853824484770112, 0.6485206028038051] |
| aux_vs_original/motion_only | [0, 1, 5, 0] | [-21.465514378724137, 2.548059054463529] |
| aux_vs_shuffled/full | [0, 2, 4, 0] | [-5.282753110294577, 5.09206633669359] |
| aux_vs_shuffled/motion_only | [0, 2, 4, 0] | [-5.606678915491146, 3.2158133684806716] |
| control_vs_original/full | [0, 0, 6, 0] | [0.0, 0.0] |
| control_vs_original/motion_only | [0, 0, 6, 0] | [0.0, 0.0] |
| shuffled_vs_control/full | [1, 1, 4, 0] | [-12.021802436361192, 1.2518199789103877] |
| shuffled_vs_control/motion_only | [1, 1, 4, 0] | [-25.364870317061452, 6.224986275805177] |

## All Primary Intervals

| Contrast / family / assignment | Point (%) | 95% locality CI |
|---|---:|---|
| aux_vs_control/full/producer0_controller1 | 0.20508068652669748 | [0.03906807138603176, 0.36764131121253374] |
| aux_vs_control/full/producer0_controller2 | -14.853824484770112 | [-29.170591968493454, -0.5370570010467716] |
| aux_vs_control/full/producer1_controller0 | 0.6485206028038051 | [-2.215201926135866, 4.120288748219221] |
| aux_vs_control/full/producer1_controller2 | -1.700533596699037 | [-5.0356893489631664, 0.03384737356329293] |
| aux_vs_control/full/producer2_controller0 | -2.3161591866929747 | [-6.3026285860307585, 0.14530518790944158] |
| aux_vs_control/full/producer2_controller1 | -3.905289619918652 | [-10.723040682956324, 0.6673701410418372] |
| aux_vs_control/motion_only/producer0_controller1 | -0.07305932851405694 | [-1.1476985535342616, 0.9461297159933235] |
| aux_vs_control/motion_only/producer0_controller2 | -21.465514378724137 | [-73.22496084660673, 8.67115240418236] |
| aux_vs_control/motion_only/producer1_controller0 | -5.60781038223659 | [-18.526769967845087, 1.5399031862288401] |
| aux_vs_control/motion_only/producer1_controller2 | -13.292541674894439 | [-39.83656868520167, -0.007904975039241858] |
| aux_vs_control/motion_only/producer2_controller0 | -5.275008077602652 | [-16.23711787234254, 0.33595641770175866] |
| aux_vs_control/motion_only/producer2_controller1 | 2.548059054463529 | [-1.697420676152607, 9.290839187536763] |
| aux_vs_original/full/producer0_controller1 | 0.20508068652669748 | [0.03906807138603176, 0.36764131121253374] |
| aux_vs_original/full/producer0_controller2 | -14.853824484770112 | [-29.170591968493454, -0.5370570010467716] |
| aux_vs_original/full/producer1_controller0 | 0.6485206028038051 | [-2.215201926135866, 4.120288748219221] |
| aux_vs_original/full/producer1_controller2 | -1.700533596699037 | [-5.0356893489631664, 0.03384737356329293] |
| aux_vs_original/full/producer2_controller0 | -2.3161591866929747 | [-6.3026285860307585, 0.14530518790944158] |
| aux_vs_original/full/producer2_controller1 | -3.905289619918652 | [-10.723040682956324, 0.6673701410418372] |
| aux_vs_original/motion_only/producer0_controller1 | -0.07305932851405694 | [-1.1476985535342616, 0.9461297159933235] |
| aux_vs_original/motion_only/producer0_controller2 | -21.465514378724137 | [-73.22496084660673, 8.67115240418236] |
| aux_vs_original/motion_only/producer1_controller0 | -5.60781038223659 | [-18.526769967845087, 1.5399031862288401] |
| aux_vs_original/motion_only/producer1_controller2 | -13.292541674894439 | [-39.83656868520167, -0.007904975039241858] |
| aux_vs_original/motion_only/producer2_controller0 | -5.275008077602652 | [-16.23711787234254, 0.33595641770175866] |
| aux_vs_original/motion_only/producer2_controller1 | 2.548059054463529 | [-1.697420676152607, 9.290839187536763] |
| aux_vs_shuffled/full/producer0_controller1 | -0.10559672420607175 | [-0.5107514851692203, 0.1844250566218617] |
| aux_vs_shuffled/full/producer0_controller2 | -2.3100705560920076 | [-4.97067712911168, -0.27976093096858345] |
| aux_vs_shuffled/full/producer1_controller0 | 1.1937837231718225 | [-0.09741046258142087, 2.874950550642875] |
| aux_vs_shuffled/full/producer1_controller2 | 5.09206633669359 | [-0.01402561225271886, 13.873561268764622] |
| aux_vs_shuffled/full/producer2_controller0 | -1.3805853485462312 | [-4.708151167385362, 1.9957069360244124] |
| aux_vs_shuffled/full/producer2_controller1 | -5.282753110294577 | [-11.230964831914399, -1.0928589418493067] |
| aux_vs_shuffled/motion_only/producer0_controller1 | -5.606678915491146 | [-15.783016613016, -0.17761398141616488] |
| aux_vs_shuffled/motion_only/producer0_controller2 | -0.8058000728981831 | [-12.441019944860649, 10.010670398924688] |
| aux_vs_shuffled/motion_only/producer1_controller0 | -2.5611960314205255 | [-8.557321797574728, 0.7731053337317415] |
| aux_vs_shuffled/motion_only/producer1_controller2 | -2.343761477557424 | [-8.23749193264088, 1.2006737916117523] |
| aux_vs_shuffled/motion_only/producer2_controller0 | 3.2158133684806716 | [-0.4562108900868454, 6.887837627048189] |
| aux_vs_shuffled/motion_only/producer2_controller1 | -4.185563881213552 | [-10.525457362003145, -0.07668733354477618] |
| control_vs_original/full/producer0_controller1 | 0.0 | [0.0, 0.0] |
| control_vs_original/full/producer0_controller2 | 0.0 | [0.0, 0.0] |
| control_vs_original/full/producer1_controller0 | 0.0 | [0.0, 0.0] |
| control_vs_original/full/producer1_controller2 | 0.0 | [0.0, 0.0] |
| control_vs_original/full/producer2_controller0 | 0.0 | [0.0, 0.0] |
| control_vs_original/full/producer2_controller1 | 0.0 | [0.0, 0.0] |
| control_vs_original/motion_only/producer0_controller1 | 0.0 | [0.0, 0.0] |
| control_vs_original/motion_only/producer0_controller2 | 0.0 | [0.0, 0.0] |
| control_vs_original/motion_only/producer1_controller0 | 0.0 | [0.0, 0.0] |
| control_vs_original/motion_only/producer1_controller2 | 0.0 | [0.0, 0.0] |
| control_vs_original/motion_only/producer2_controller0 | 0.0 | [0.0, 0.0] |
| control_vs_original/motion_only/producer2_controller1 | 0.0 | [0.0, 0.0] |
| shuffled_vs_control/full/producer0_controller1 | 0.3089113652148727 | [0.03304825697052791, 0.7642816053582] |
| shuffled_vs_control/full/producer0_controller2 | -12.021802436361192 | [-25.988602597491557, -0.255370921777991] |
| shuffled_vs_control/full/producer1_controller0 | -0.5371748782422222 | [-3.2615751128248287, 1.5124359226604984] |
| shuffled_vs_control/full/producer1_controller2 | -8.469860634138563 | [-23.858585660010238, 0.047660934321203205] |
| shuffled_vs_control/full/producer2_controller0 | -0.9745450362272711 | [-3.510070799984864, 1.5119999041243752] |
| shuffled_vs_control/full/producer2_controller1 | 1.2518199789103877 | [-0.14956163922384325, 3.447657528546636] |
| shuffled_vs_control/motion_only/producer0_controller1 | 1.8887478019445192 | [-0.22517078975030375, 5.73187892868728] |
| shuffled_vs_control/motion_only/producer0_controller2 | -25.364870317061452 | [-76.53241117738824, 0.3570638252312163] |
| shuffled_vs_control/motion_only/producer1_controller0 | -1.8934718301218 | [-6.508215579655921, 0.7649591903186265] |
| shuffled_vs_control/motion_only/producer1_controller2 | -11.346969736954623 | [-32.74809530899504, -0.013441201262195814] |
| shuffled_vs_control/motion_only/producer2_controller0 | -10.869329275035739 | [-24.129015780569567, 0.5275524468142916] |
| shuffled_vs_control/motion_only/producer2_controller1 | 6.224986275805177 | [0.12736207955791434, 12.32261047205244] |

Three seeds averaged within locality,3000 paired resamples of four localities per assignment.
Assignments overlap; previously exposed source development. No multiplicity-adjusted or independent claim.
Original native inputs,GELU64,four-cost loss,all-known support and site-balanced sampling are retained.
Only the auxiliary objective changes among new arms. Reference outputs remain frozen for held cost comparisons.
The event is producer-relative cap exceedance,not generic easy membership. Inner/outer producer transport remains.
Full/motion changes forecasts and populations,not a matched feature ablation. No threshold/model selection.
Training diagnosis uses dependent views and is not causal proof. Legacy original fitting subset is unavailable.

```json
{
  "primary_cost_gate": false,
  "tail_coverage_all_harm_guards": false,
  "true_vs_shuffled_gate": false,
  "auxiliary_cost_contribution": false,
  "new_policy_evaluated": false,
  "independent_confirmation": false,
  "deployment_changed": false,
  "submission_ready": false,
  "stage5c_executed": false,
  "smc_enabled": false,
  "strong_control_reconstructed": true
}
```

Obs8/pred12 annotation steps,detector pixels. No metric/seconds,physical-safety,human-gold,true3D or foundation claims.
