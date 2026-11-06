# Geometric Actor Result And Remaining Gap

Primary known:-5.1331pp;paired lower:-21.4474pp.
Component screen:False. Camera-CV effect:+2.8275known pp.
Neural minus camera-CV:+7.4171known pp.

These controls distinguish simple geometric forecasting from additional
learned forecasting. They do not isolate true camera pose,depth,scene
semantics or the causes of every source failure. Constant-affine future
extrapolation can fail under changing viewpoint,parallax or measurement
error. Those are hypotheses unless the retained slices support them.

A negative completion bound alone does not prove the unknown effect is
negative. Known,partial and wholly unknown outcomes stay distinct.
The source-fold contrasts and camera support slices remain visible.
All source rows remain;no primary/threshold is changed after scoring.

## Fitting-Side Diagnostic

The camera actor has higher final fitting-minibatch loss in6/6designs.
The original draws,factors and final batch are matched. This is one
final minibatch per design,not a complete training-distribution average
or proof of underfitting. The degradation already appears on fitting
examples,so a pure held-source-shift explanation is insufficient.

| Design | Identity final loss | Camera final loss |
|---|---:|---:|
| eqmotion_A0_inner0_seed17 | 0.663961 | 0.755396 |
| eqmotion_A0_inner1_seed17 | 0.442291 | 0.608595 |
| eqmotion_A0_inner2_seed17 | 0.532024 | 0.641094 |
| eqmotion_A1_inner0_seed17 | 0.619711 | 0.717687 |
| eqmotion_A1_inner1_seed17 | 0.705646 | 0.831796 |
| eqmotion_A1_inner2_seed17 | 0.693550 | 0.760640 |

## Next Model-Side Question

The current intervention bundles past-history alignment with fixed
future-affine restoration. Their separate predictive effects are not
identified. A next matched A-side ablation should separate those two
operations before scaling the same failed recipe or retuning a policy.
It must retain source exclusions,original loss/risk/support rules and
a native-coordinate control. No new test is opened by this report.

No new controller is trained and no new policy is deployed. The original
per-recording2%all/easy,signed easy-net,reference/support/NA requirements
remain. Exact replay is reproducibility evidence,not independent
confirmation. Three forecaster seeds and compliant new confirmation
remain missing. Prior image-feature and prefix-loss failures are not
overwritten. The paper is not submission-ready.
