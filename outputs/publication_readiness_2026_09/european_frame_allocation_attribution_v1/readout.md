# Frozen-query neural use and harm attribution

## Material Passport

Registered after the main readout,before these new attribution values. Post-readout diagnostic,not a primary efficacy test.
Fresh inference and arithmetic on cached_verified models,inputs and immutable action choices.
All72 original independent attributions and288 action hashes match. Separate full diagnostic replay is exact.
PID25357 takes101.952s;replay PID26550 takes118.479s.
SOURCE fitting/resubstitution only. No training,policy search,independent confirmation or deployment.

## Actual neural use within each frozen allocation

Delete ONLY selected neural actions to the unchanged floor;keep all other actions fixed.
This is NOT a reoptimized physical-only controller. The ablation itself need not satisfy
the original budget,because removing a reference-contributing action can change the ratio.

|Frozen decoder|Known neural-use increment pp [95% CI]|Wholly-unknown completion-lower pp [95% CI]|
|---|---:|---:|
|independent|+0.657318 [+0.189644,+1.302010]|+0.585409 [+0.117586,+1.242017]|
|frame_uniform|+0.280065 [+0.056973,+0.649359]|+0.249833 [+0.013924,+0.632573]|
|joint_matched|+1.026140 [+0.297522,+1.991966]|+0.905772 [+0.186673,+1.874977]|
|joint_free|+1.080669 [+0.324516,+2.080735]|+0.958757 [+0.212395,+1.962871]|

## Where the joint-minus-independent gain comes from

Incremental neural-use reallocation:known+0.368822 [+0.104733,+0.710547];completion-lower+0.311363 [+0.063381,+0.646258].
Remaining action reallocation:known+0.279566 [+0.075391,+0.561234];completion-lower+0.154185 [-0.046914,+0.426301].
The known point estimates sum to the main+0.648388pp increment. Interval endpoints do not sum.
Shared unknown neural uses cancel before bounding the difference. The incremental lower
bound is not a subtraction of two lower bounds. These are forecast-use arithmetic effects,
not causal effects on real agent behavior,new dynamics training or independent neural superiority.

## Why easy protection still fails

|Decoder|Observed easy harm bookkeeping|Predicted easy harm bookkeeping|Median within-record predicted/observed easy harm|Fully known selected queries with easy excess|
|---|---:|---:|---:|---:|
|independent|19255.862315|254.215456|0.014333063206754988|18590|
|frame_uniform|1062.944833|9.396859|0.007316757326994352|1530|
|joint_matched|33267.345219|429.765075|0.01520479888335473|18324|
|joint_free|34020.866431|691.534622|0.019952793093846146|19123|

Joint-matched fully known query witnesses:{'numerator_only': 17728, 'denominator_only': 32, 'both_single_replacements_fail': 326, 'only_joint_replacement_fails': 238}.
Selected-unknown query views:3294;empty query views:50332.
Pooling deliberately permits individual predicted row excess. Such a row is not necessarily
a false query admission; only complete-query arithmetic on known selected outcomes is used
for that diagnosis. Mixed/unknown queries remain in full policy scores and are not imputed.
The easy-harm estimate is much too small on the selected population,even while total harm
can be conservative in aggregate. Fewer positive-excess rows do not imply smaller harm mass.
Witnesses describe numerator/reference errors algebraically,not a uniquely identified causal root cause.

## Interpretation and boundaries

Raw masses sum reused local-image-coordinate views and are bookkeeping,not cross-domain physical metrics.
Within-record ratios exclude records with zero observed harm;their quantiles are not a calibration guarantee.
Intervals are3,000 locality bootstrap draws after context/head-seed averaging on12 exposed localities.
Three head seeds are not three forecaster seeds or independent scenes. The main complete-policy
risk gate still fails:324known easy violations,43/444 supported recording views. No positive
secondary effect can replace it. Independent confirmation and submission readiness remain false.
The next strong control must reoptimize physical-only allocation at a feasible common query
count,not rename neural deletion as that comparator. Selected-query easy-risk estimation then
needs a distinct,evidence-driven repair and exposure-aware calibration protocol.
No metric/seconds/human-gold/true3D/foundation claim,Stage5C execution or SMC.
