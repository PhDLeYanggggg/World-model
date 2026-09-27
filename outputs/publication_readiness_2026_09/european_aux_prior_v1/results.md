# Fitting Cap-Prior Intercept Repair

fresh_run:288 native Torch heads,576000 optimizer updates,144 exposed source-held readouts.
cached_verified:original cost-only,true/shuffled controls,forecasts,features and nested labels.
not_run:new forecasting or policy,independent selection,reserved calibration,confirmation.

| Contrast / family | Positive / negative / overlap / missing CIs | Primary point range (%) |
|---|---|---|
| repair_vs_cost_only/full | [1, 2, 3, 0] | [-12.412007766759519, 0.9243409620939729] |
| repair_vs_cost_only/motion_only | [0, 3, 3, 0] | [-10.51611653293162, 0.11600505512846675] |
| repair_vs_old_true/full | [2, 0, 4, 0] | [-10.748401218312146, 5.645867901921571] |
| repair_vs_old_true/motion_only | [0, 1, 5, 0] | [-11.258706929601615, 6.153141884483629] |
| repair_vs_matched_shuffled/full | [0, 1, 5, 0] | [-11.8103279342166, 0.013760566935088106] |
| repair_vs_matched_shuffled/motion_only | [0, 1, 5, 0] | [-8.488024240557694, 0.16114034164670826] |
| shuffled_repair_vs_old_shuffled/full | [2, 2, 2, 0] | [-1.6256071877746106, 6.720702541302116] |
| shuffled_repair_vs_old_shuffled/motion_only | [1, 2, 3, 0] | [-13.598735280711256, 5.93326962564037] |
| old_true_vs_cost_only/full | [1, 1, 4, 0] | [-14.853824484770112, 0.6485206028038051] |
| old_true_vs_cost_only/motion_only | [0, 1, 5, 0] | [-21.465514378724137, 2.548059054463529] |

## All Primary Intervals

| Contrast / family / assignment | Point (%) | 95% locality CI |
|---|---:|---|
| repair_vs_cost_only/full/producer0_controller1 | -0.1853261260268881 | [-0.714456243762931, 0.3438039917091548] |
| repair_vs_cost_only/full/producer0_controller2 | -7.980546127229713 | [-15.937011196571369, -0.02408105788805849] |
| repair_vs_cost_only/full/producer1_controller0 | 0.9243409620939729 | [0.04005809336686421, 1.8086238308210816] |
| repair_vs_cost_only/full/producer1_controller2 | -0.2725887071605391 | [-1.1753036680796674, 0.29696013763539886] |
| repair_vs_cost_only/full/producer2_controller0 | -12.412007766759519 | [-32.16047073691381, 0.4280397289285406] |
| repair_vs_cost_only/full/producer2_controller1 | -11.557824155531721 | [-22.52023005510964, -0.5954182559538019] |
| repair_vs_cost_only/motion_only/producer0_controller1 | -10.51611653293162 | [-30.612821441389194, -0.11527985124363005] |
| repair_vs_cost_only/motion_only/producer0_controller2 | -10.469323170926625 | [-30.037901870120177, -0.2804121918685743] |
| repair_vs_cost_only/motion_only/producer1_controller0 | 0.11600505512846675 | [-1.9838969387191576, 2.142226428316044] |
| repair_vs_cost_only/motion_only/producer1_controller2 | -2.284954586867556 | [-6.810892775344861, 0.00282558510408329] |
| repair_vs_cost_only/motion_only/producer2_controller0 | -3.1421466507224096 | [-7.1980119673440015, -0.21583516512612838] |
| repair_vs_cost_only/motion_only/producer2_controller1 | -3.629587483790825 | [-15.805396721491569, 4.898465142042753] |
| repair_vs_old_true/full/producer0_controller1 | -0.3934392363893567 | [-1.0062965568045765, 0.21941808402586316] |
| repair_vs_old_true/full/producer0_controller2 | 5.645867901921571 | [0.5021215412804136, 11.15131695236551] |
| repair_vs_old_true/full/producer1_controller0 | 0.05568331692280436 | [-2.925238362525634, 3.095735723297217] |
| repair_vs_old_true/full/producer1_controller2 | 1.3326346383518133 | [0.026443550154782502, 3.5789898386771846] |
| repair_vs_old_true/full/producer2_controller0 | -10.748401218312146 | [-27.523636424246142, 0.24695703329754115] |
| repair_vs_old_true/full/producer2_controller1 | -10.048890006365198 | [-29.80962407751152, 0.607897948988636] |
| repair_vs_old_true/motion_only/producer0_controller1 | -11.258706929601615 | [-33.86639482522224, 0.18801912399586507] |
| repair_vs_old_true/motion_only/producer0_controller2 | -0.598826548743693 | [-17.29456259938539, 16.20083711322673] |
| repair_vs_old_true/motion_only/producer1_controller0 | 0.6826485616740595 | [-3.716453724371245, 5.738225659225008] |
| repair_vs_old_true/motion_only/producer1_controller2 | 6.153141884483629 | [-0.014911217315748414, 18.463609662263643] |
| repair_vs_old_true/motion_only/producer2_controller0 | 1.168131938636304 | [-2.5689926257765765, 6.367547543525793] |
| repair_vs_old_true/motion_only/producer2_controller1 | -6.818012601633807 | [-13.602485416957599, -0.03353978631001593] |
| repair_vs_matched_shuffled/full/producer0_controller1 | 0.013760566935088106 | [-0.309802821632593, 0.33732395550276917] |
| repair_vs_matched_shuffled/full/producer0_controller2 | -3.331736078762024 | [-11.351206698735368, 1.1645960277453287] |
| repair_vs_matched_shuffled/full/producer1_controller0 | -0.04612814314470336 | [-2.7251587406744795, 2.765437483588491] |
| repair_vs_matched_shuffled/full/producer1_controller2 | -0.7695160793082189 | [-2.3851790399176322, 0.08119381243351115] |
| repair_vs_matched_shuffled/full/producer2_controller0 | -11.8103279342166 | [-30.716059699248568, 0.1422106416701172] |
| repair_vs_matched_shuffled/full/producer2_controller1 | -9.857220526872794 | [-19.231541512038206, -0.48289954170738414] |
| repair_vs_matched_shuffled/motion_only/producer0_controller1 | -8.488024240557694 | [-23.410366381123076, -0.1986334973321402] |
| repair_vs_matched_shuffled/motion_only/producer0_controller2 | -3.029068631545656 | [-9.298470781323468, 0.7420202286319695] |
| repair_vs_matched_shuffled/motion_only/producer1_controller0 | 0.16114034164670826 | [-1.5939446257550336, 1.8165788927129238] |
| repair_vs_matched_shuffled/motion_only/producer1_controller2 | -0.3756727205504775 | [-1.4719679485325072, 0.3368219124082504] |
| repair_vs_matched_shuffled/motion_only/producer2_controller0 | -0.7186736538528962 | [-3.8074804375004536, 1.867944285795844] |
| repair_vs_matched_shuffled/motion_only/producer2_controller1 | -5.537475746114341 | [-18.943891504339398, 2.3068294261630147] |
| shuffled_repair_vs_old_shuffled/full/producer0_controller1 | -0.5147685257848645 | [-0.9389144468866746, -0.09062260468305457] |
| shuffled_repair_vs_old_shuffled/full/producer0_controller2 | 5.492056727167492 | [0.037090798308554065, 10.94702265602643] |
| shuffled_repair_vs_old_shuffled/full/producer1_controller0 | 1.199087497068197 | [0.07576019374389957, 2.445994496307071] |
| shuffled_repair_vs_old_shuffled/full/producer1_controller2 | 6.720702541302116 | [-0.06844997565672935, 18.36755121931045] |
| shuffled_repair_vs_old_shuffled/full/producer2_controller0 | 0.2674295807398367 | [-1.5948574058383507, 1.8253623837267186] |
| shuffled_repair_vs_old_shuffled/full/producer2_controller1 | -1.6256071877746106 | [-4.186318175269673, -0.056802520485120134] |
| shuffled_repair_vs_old_shuffled/motion_only/producer0_controller1 | -13.598735280711256 | [-41.941008302544525, 1.221577996011801] |
| shuffled_repair_vs_old_shuffled/motion_only/producer0_controller2 | 2.1107902408214576 | [-3.5368768671809065, 10.02231243162553] |
| shuffled_repair_vs_old_shuffled/motion_only/producer1_controller0 | -3.920036176299876 | [-7.705693444334622, -0.1343789082651304] |
| shuffled_repair_vs_old_shuffled/motion_only/producer1_controller2 | 5.93326962564037 | [0.00813136348240175, 16.939858752668613] |
| shuffled_repair_vs_old_shuffled/motion_only/producer2_controller0 | 4.988725929665984 | [-0.5343455248109821, 10.511797384142948] |
| shuffled_repair_vs_old_shuffled/motion_only/producer2_controller1 | -6.273297855880848 | [-12.846776260806042, -0.13468193005686838] |
| old_true_vs_cost_only/full/producer0_controller1 | 0.20508068652669748 | [0.03906807138603176, 0.36764131121253374] |
| old_true_vs_cost_only/full/producer0_controller2 | -14.853824484770112 | [-29.170591968493454, -0.5370570010467716] |
| old_true_vs_cost_only/full/producer1_controller0 | 0.6485206028038051 | [-2.215201926135866, 4.120288748219221] |
| old_true_vs_cost_only/full/producer1_controller2 | -1.700533596699037 | [-5.0356893489631664, 0.03384737356329293] |
| old_true_vs_cost_only/full/producer2_controller0 | -2.3161591866929747 | [-6.3026285860307585, 0.14530518790944158] |
| old_true_vs_cost_only/full/producer2_controller1 | -3.905289619918652 | [-10.723040682956324, 0.6673701410418372] |
| old_true_vs_cost_only/motion_only/producer0_controller1 | -0.07305932851405694 | [-1.1476985535342616, 0.9461297159933235] |
| old_true_vs_cost_only/motion_only/producer0_controller2 | -21.465514378724137 | [-73.22496084660673, 8.67115240418236] |
| old_true_vs_cost_only/motion_only/producer1_controller0 | -5.60781038223659 | [-18.526769967845087, 1.5399031862288401] |
| old_true_vs_cost_only/motion_only/producer1_controller2 | -13.292541674894439 | [-39.83656868520167, -0.007904975039241858] |
| old_true_vs_cost_only/motion_only/producer2_controller0 | -5.275008077602652 | [-16.23711787234254, 0.33595641770175866] |
| old_true_vs_cost_only/motion_only/producer2_controller1 | 2.548059054463529 | [-1.697420676152607, 9.290839187536763] |

## Interpretation Boundary
Primary is expected easy-harm cost MSE on positive envelopes,not trajectory ADE/FDE or deployment lift.
Three seeds averaged within locality;3000 paired four-locality resamples per assignment.
Six assignments overlap and reuse previously exposed source development;intervals are descriptive and unadjusted.
All 2000-update final checkpoints were frozen before readout. No checkpoint,threshold or role selection.
Only auxiliary intercept changed. Objective,shared/cost initialization,draws,targets and budgets are matched.
Source-held reference costs are frozen. Auxiliary probabilities never multiply expected harm.
Full/motion families also change forecasts/event populations;they are not matched feature ablations.
Fitting secondary summaries average dependent fitting views,not an independent generalization test.

```json
{
  "primary_cost_gate": false,
  "tail_coverage_all_harm_guards": false,
  "true_vs_shuffled_gate": false,
  "repair_advance_gate": false,
  "independent_confirmation": false,
  "deployment_changed": false,
  "new_policy_evaluated": false,
  "submission_ready": false,
  "stage5c_executed": false,
  "smc_enabled": false
}
```

Obs8/pred12 native annotation steps,detector pixels. No metric/seconds,physical-safety,human-gold,true3D or foundation claims.
