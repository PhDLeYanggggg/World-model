# Frozen Utility / Variable Guard: Complete Bounded Repair

## Material Passport

Mode:run and numerical reproducibility validation. Fresh_run:one registered
decision repair,nine frozen five-policy action matrices and54metric groups,
plus separate choice/metric replay. Cached_verified:all27SOURCE-trained cost
critics and their outputs,original forecasts/inputs and exposed selection labels.
No new training,trajectory inference,cost inference or threshold search.

## Why This Was Tested

The preceding actual-seed disagreement experiment lost utility while reducing
some harm. Its shared input changed both ranking and guards. The causal-only
audit finds30,135repeated candidate views with old predicted utility>0,new real
guards passing,but new predicted utility<=0. These are predicted opportunities,
not oracle useful actions. This preregistered repair tests exactly one inference
factor rather than training another model or sweeping thresholds.

Keep the preceding disabled mean_absolute B-H utility/ranking fixed. Use real
or noise H/R and EH/ER guards at the unchanged2%. Choose the largest old positive
utility passing those guards,otherwise keep the same SOURCE floor. Guard benefit
is not used. The two learned quantities need not define a single coherent
posterior;recombination is not a calibration or safety theorem. All three
coupled controls must reproduce the preceding experiment exactly.

## Registered Contrasts

Primary:fixed-real minus fixed-noise paired completion lower. Advancement:
fixed-real minus coupled-disabled. Mechanism secondary:fixed-real minus coupled-real.

|Contrast|Known secondary,pp [95% CI]|Paired completion lower,pp [95% CI]|
|---|---:|---:|
|fixed_real_minus_coupled_disabled|-0.95798 [-1.48914, -0.42528]|-1.43668 [-2.04965, -0.84298]|
|fixed_real_minus_coupled_real|-0.03985 [-0.06930, -0.01135]|-0.12084 [-0.18955, -0.06826]|
|fixed_real_minus_fixed_noise|-0.65204 [-1.16011, -0.24558]|-1.01745 [-1.56067, -0.63740]|

Primary pass:**False**. Advancement pass:**False**.
Original full-record risk:**False**.
Candidate:**False**. Deployment changed:**false**.
Improving over a failed coupled policy alone does not meet advancement.

## Complete Policy And Risk Results

|Policy|Known ADE gain vs same floor,% [CI]|Mean intervention|Worst easy net|Supported repeated record views|Known all/easy risk violations|Undefined easy risk|
|---|---:|---:|---:|---:|---:|---:|
|coupled_disabled|+1.50375 [+0.81699, +2.13074]|22.496%|8.495%|6/252|199/211|14|
|coupled_real|+0.58562 [+0.34600, +0.85991]|13.452%|6.870%|13/252|188/178|27|
|coupled_noise|+1.19709 [+0.61950, +1.81348]|20.870%|8.779%|8/252|208/209|13|
|fixed_utility_real_guard|+0.54577 [+0.31860, +0.80328]|14.706%|7.848%|15/252|180/179|25|
|fixed_utility_noise_guard|+1.19781 [+0.64395, +1.79289]|20.745%|8.609%|8/252|207/212|14|

All38,102targets/28recordings/six exposed selection localities remain,including
848wholly unknown and15,820partially labelled outcomes. Original all/easy
positive-harm and easy-net2%budgets apply to every required recording. Benefits
do not cancel positive-harm numerators. Empty selected-easy reference is not
estimable,not safe zero.252views repeat28recordings across nine producer models.

Bootstrap3000localities after averaging SOURCEfolds/seeds within each of six
localities. Overlapping rows are not independent observations. Completion
bounds apply to wholly unknown rows under available-label ADE;they do not fill
missing steps of partial trajectories. Full-roster endpoint FDE is not identified.
No same-count comparison or joint-allocation contribution is claimed here.

## Execution And Reproduction

Decision freeze PID232:13.410177s,peak2201534464B.
Evaluation PID840:16.971586s,peak2327592960B.
Separate verification PID1121:16.747841s. All nine complete action
matrices and54metric groups reproduce exactly. Cached critic costs are checked
by hash,not freshly inferred in this repair. The preceding27-model Torch replay
is a separate inherited receipt. Private arrays/weights remain outside Git.

```sh
.venv-pytorch/bin/python scripts/run_m3w_frozen_utility_guard.py register
.venv-pytorch/bin/python scripts/run_m3w_frozen_utility_guard.py freeze
.venv-pytorch/bin/python scripts/run_m3w_frozen_utility_guard.py eval
.venv-pytorch/bin/python scripts/run_m3w_frozen_utility_guard.py verify
.venv-pytorch/bin/python scripts/report_m3w_frozen_utility_guard.py
```

Completed endpoints are immutable;partial freeze/eval supports--resume. Existing
scalar reports and tests can be rerun without replacing completed phase receipts.
Local arm64CPU4/workers0,12hboundary/10GiBreserve;no CREATE submission.

## Scope And Limits

This is bounded model-selection development after exposed parent results,not
an untouched final test. SOURCE models,normalizers and priors remain frozen;
calibration/confirmation/DroneCrowd stay closed. Neither a positive secondary,
extra switching nor a replay pass constitutes deployment or a new world model.
No new risk-only head was trained,so this cannot establish that separate
supervision would succeed. The old coupled negatives and matched-joint failure
remain unchanged. Main PDF was not regenerated for this repair.

obs8/pred12/rawstride12,local-image pixel,detector-silver,available labels,
2.5D/pseudo-3D trajectory state. Not old rawt50,metric,seconds,human gold,true3D
or foundation. Stage5C/SMC remain off;submission readiness false.
