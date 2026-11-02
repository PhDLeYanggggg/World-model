# SOURCE-To-Selection Cost Controllers: Complete Readout

## Result Source And Scope

Fresh_run: nine SOURCE forecast caches,18real Torch cost fits, two first-producer
complete refits, new selection cost inference, causal allocation and54metric
groups. Cached_verified: actual source-trained forecasting checkpoints, original
SOURCE whitelist, frozen selection inputs/forecasts and previously verified
model-selection labels. No new trajectory model was trained. The six selection
localities were already exposed by the supporting forecast readout, not an
untouched final test. Risk calibration, confirmation and DroneCrowd stay closed.

Every cost model fits only the eight SOURCE localities outside its forecaster's
four training localities. Fold costs use91,562/145,422/62,036rows from the original
149,510-row whitelist, repeated across actual forecasting seeds17/29/43.
The same rows are not independent new examples. No selection labels train the
weights, normalizers or thresholds. All38,102selection targets/28recordings/six
localities remain, including futureless and partially labelled targets.

## Controlled Method

Both29,253-parameter critics share initialization, complete query draws and
physical-only SOURCE feature normalization. Inputs are380causal geometry/rollout
and64past detector fields. Only the parameter-free mean of other same-query
target representations differs. Original partial-neighbor context remains in
both arms. No scene ID, raw future endpoint, central velocity, future support,
test statistic or test goal enters inference. Cost is benefit/positive-harm/
reference/easy-reference/easy-harm, not hard best-baseline classification.

The decoder shares references across actions and preserves B<=R,ER<=R,EH<=H and
B+H<=causal disagreement. Known-only underharm/ranking objectives reuse existing
cost learning. Grouped risk pooling reuses the existing allocator; this is not
a newly invented deferral loss, new JEPA/dynamics, pairwise interaction
optimization or a claim of physical safety. The physical-bank comparison
removes the neural action from the same critic, not a retrained ablation.

## Registered Primary

Context matched-joint minus context independent, relative to the same fixed
SOURCE-selected floor. Intervention counts match within every query.

|Contrast|Effect, percentage points [95% locality bootstrap CI]|
|---|---:|
|Registered paired completion-lower primary|+0.0016 [-0.0152, +0.0168]|
|Known-label secondary|+0.0138 [-0.0081, +0.0344]|
|Paired completion-upper diagnostic|+0.0259 [-0.0024, +0.0596]|
|Context minus row matched-joint, known secondary|-0.0012 [-0.0877, +0.0916]|
|Context minus row matched-joint, completion lower|-0.2111 [-0.3463, -0.0959]|
|Neural-action bank minus physical bank, known secondary|+0.0575 [+0.0320, +0.0775]|
|Neural-action bank minus physical bank, completion lower|+0.0171 [-0.0196, +0.0470]|

Primary pass: **False**. Original complete-record risk pass: **False**.
Candidate gate: **False**. Deployment changed: **false**.
No favorable secondary, free-count result or seed replaces the registered primary.

## Complete Accuracy, Intervention And Risk

|Policy|Known available-label ADE gain vs floor,% [CI]|Mean switch rate|Worst known easy net degradation|Supported repeated record views|Known all/easy risk violations|Undefined selected-easy views|
|---|---:|---:|---:|---:|---:|---:|
|context_frame_uniform|+0.1014 [+0.0364, +0.1702]|1.57%|3.07%|35/252|104/88|100|
|context_frame_uniform_physical|+0.0758 [+0.0226, +0.1370]|0.74%|3.07%|36/252|100/74|116|
|context_independent|+0.1595 [+0.0580, +0.2726]|5.52%|3.77%|23/252|147/154|52|
|context_independent_physical|+0.1131 [+0.0338, +0.2038]|2.50%|4.09%|13/252|163/152|67|
|context_joint_free|+0.2173 [+0.1058, +0.3596]|6.92%|6.29%|19/252|155/159|52|
|context_joint_free_physical|+0.1395 [+0.0479, +0.2482]|3.18%|5.97%|14/252|167/151|65|
|context_joint_matched|+0.1733 [+0.0838, +0.2708]|5.52%|7.02%|21/252|157/156|52|
|context_joint_matched_physical|+0.1158 [+0.0421, +0.1990]|2.50%|4.61%|14/252|163/150|68|
|floor|+0.0000 [+0.0000, +0.0000]|0.00%|0.00%|0/252|0/0|252|
|raw_history|+11.6729 [+9.0955, +14.1328]|100.00%|47.98%|0/252|249/247|0|
|row_frame_uniform|+0.0168 [+0.0043, +0.0320]|1.28%|1.71%|40/252|87/84|111|
|row_independent|+0.1492 [+0.1039, +0.2016]|7.40%|4.97%|21/252|165/170|42|
|row_joint_free|+0.2382 [+0.1713, +0.3320]|9.63%|7.09%|18/252|174/170|41|
|row_joint_matched|+0.1745 [+0.1334, +0.2187]|7.40%|6.57%|21/252|168/170|42|

The floor is each producer's SOURCE-selected fixed reference, not an oracle or
a baseline picked on these scenes. Raw_history and floor controls exactly match
the earlier complete selection readout. Rates average locality/fold/seed views,
not row-weighted rates. There are252repeated recording-risk views, not252unique
recordings. Positive harm cannot be canceled by benefit in the risk numerator.
The original all/easy positive-harm and easy-net budgets stay2% on every record.
The floor adds no harm; an empty selected-easy reference is not estimable, not
proof that abstention is harmful or a successful positive learned intervention.
Nonempty count-matched utility controls: 216/252views. Empty comparisons
are labelled not_estimable and do not certify safety. Full-policy floor-versus-
floor utility can be exactly zero without proving any selection mechanism.

## Locality And Seed Effects

|Selection locality|Primary known component,pp|Primary paired lower,pp|
|---|---:|---:|
|eu-locality-087|+0.05|+0.01|
|eu-locality-092|+0.04|+0.03|
|eu-locality-093|+0.01|+0.01|
|eu-locality-103|-0.00|-0.01|
|eu-locality-104|+0.01|+0.00|
|eu-locality-125|-0.03|-0.03|

|Actual forecasting/training pipeline seed|Mean primary known component,pp|Mean primary paired lower,pp|
|---|---:|---:|
|17|+0.04|+0.01|
|29|+0.00|+0.00|
|43|-0.00|-0.01|

Intervals use3000locality resamples after averaging source folds/seeds within
each of six localities. Repeated forecasts/windows are not independent scenes.
These are conditional model-selection development intervals, not calibrated
population guarantees or independent confirmation. Unknown disagreement bounds
follow the previous available-label ADE convention and do not recover missing
steps of partial trajectories. Full-roster FDE is not estimable; available-endpoint
FDE in scalar groups is diagnostic. No future-completeness filter changes the roster.

## Real Training And Recovery

|Arm|Complete fits|Primary updates|Mean first minibatch loss|Mean final minibatch loss|
|---|---:|---:|---:|---:|
|row|9|70656|3.40376|0.26793|
|context|9|70656|3.40376|0.30407|

Every original fitting query has at least four visits. No unknown future has a
supervised-zero label. These are varying SOURCE minibatches, not validation losses
or generalization evidence. Complete scalar traces are exported separately.

The original400-update real pilot failed bit-exact context recovery by2.42e-08;
its log/checkpoints and original registration are preserved. Registration_v2
replaces scatter/gather reductions with fixed membership products. The repaired
real pilot and two full first-producer refits check exactness without a relaxed
tolerance. This verifies the repair on tested runs, not a unique hardware cause
or an OpenMP/SHM hang. No new policy outcomes were scored before that repair.

Formal training PID87270: 541.081s, peak RSS
14,854,356,992B. Primary updates141,312;
full-refit updates14,336. Readout PID88704:
34.578s. Separate verification PID88908: 44.865s.
All18fresh cost inferences and54frozen-action metric groups replay exactly;
query feasibility and matched counts are checked. Solver optimization is **not**
claimed reexecuted by this verification. Local arm64CPU4/workers0 was adequate;
no CREATE job was submitted and no remote scheduler state is inferred.

## Remaining Paper Requirements

Do not equate a completed program with the paper hypothesis. A useful and safe
joint policy, conflict/composition controls, independent risk calibration/final
confirmation, appropriate public predictors and anonymous end-to-end reproduction
remain necessary. Selection results can choose the next method, not serve as
an untouched final test. No deployment or submission-readiness promotion here.

obs8/pred12/rawstride12, local-image pixel, detector-silver,2.5D/pseudo-3D agent
trajectory state modeling. Not metric, seconds-level, human gold, true3D or
foundation evidence. Stage5C is not executed and SMC is not enabled.
