# Cost Forests Improve Empirical Protection, Not Established Superiority

## Evidence Status

2026-09-22. **fresh_run**: 24 standard ExtraTrees cost regressors, fixed readout,
checkpoint replay and separate numerical checks. **cached_verified**: nested
source-excluded neural forecasts, 356 causal features and the temporal neural
risk heads. **not_run**: new trajectory-predictor training, independent risk
calibration/confirmation, external confirmation and the raw-frame t+50 supplement.

The registration was committed and pushed as `1985a3bc` before real fitting.
All four SDD physical sites are development-exposed. This is the eight-observed/
twelve-predicted annotation-step task at raw stride 12, in annotation pixels.
It is not a Stage37 raw-t50 replication, metric/seconds prediction, true-3D or
foundation result. Historical Stage26/37 results are not recertified. Stage5C
and SMC are off. No model is promoted or deployed.

## What Was Trained

Two existing candidate actions, ramp and displacement-matched uniform, each
receive a forest at every four-site/three-seed view. Each forest has 128 trees,
depth at most 16, leaf minimum 64 unique rows, max_features 1/3, no bootstrap,
and a joint two-output squared-error objective on benefit/D and harm/D. Inference
multiplies the leaf-average fractions by causal forecast disagreement D.
These are expected-cost estimates, not calibrated failure probabilities.

The same fitting row support, nested upstream exclusions, train normalization,
old actual neural sampler counts and fixed region weighting are retained.
The objectives, capacities and compute are **not** matched. This is a conventional
strong comparator, not proof that one architectural component caused a change
or that all possible tree methods have been exhaustively tuned.

All 24 fits completed: **3,072 trees**, **4,161,296 nodes**, **619.66 recorded fitting
seconds**, and 424,060,784 checkpoint bytes (about 404 MiB). Effective positive-
weight fitting rows range from 67,489 to 128,572 per endpoint. Fitting time
includes in-sample trace prediction and checkpoint work, excludes upstream
forecast training, preparation, provenance hashing and held readout. Reused
neural sampler counts are weights, not new SGD draws. The native arm64 run used
four fitting threads, one prediction thread and no worker processes. A 16-tree
full-row pilot was resumed; no held outcomes selected the budget or settings.

## Fixed Results

Gain is equal-physical-site relative ADE/FDE improvement over causal CV; errors
are averaged over seeds before aggregation. Negative easy degradation means
improvement. Switch counts include repeated seed instances, not independent
trajectories. No outcomes were used to match switching counts.

| Fixed policy | ADE gain | FDE gain | Hard gain | Aggregate easy degradation | Worst site/seed easy degradation | Switches | Exact-zero-CV harmed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Ramp forest: primary** | **3.53029%** | **5.16628%** | **3.76414%** | **-1.74876%** | **-0.13424%** | **22,539** | **0** |
| Ramp neural: control | 3.39756% | 4.90029% | 3.67549% | -1.24935% | 0.91149% | 33,793 | 1 |
| Neural net-gain ranking at ramp forest count | 3.82776% | 5.28068% | 5.80729% | 1.51097% | 2.55829% | 22,539 | 0 |
| Uniform forest: secondary | 3.87323% | 4.06115% | 4.17984% | -1.09200% | 2.07770% | 24,706 | 0 |
| Uniform neural: secondary control | 3.64449% | 3.85268% | 3.95700% | -0.63522% | 2.56229% | 36,105 | 1 |
| Neural net-gain ranking at uniform forest count | 4.13218% | 4.42979% | 6.26396% | 2.98630% | 4.44135% | 24,706 | 0 |

The primary ADE difference is **+0.13273 percentage points**, with paired
3,000-physical-site bootstrap CI **[-0.06206, +0.48134] pp**. It is not an
established improvement. The four scene differences are +0.65389, -0.03634,
-0.08080 and -0.00584 pp: the positive mean comes from coupa, not broad
accuracy superiority across sites. Uniform's secondary contrast also crosses
zero, and its worst easy degradation exceeds 2%; do not promote it.

Against CV, primary gain has CI [2.25314%, 4.84074%]. At the forest's switch
count, neural net-gain ranking has 0.29747 pp greater point gain; forest-minus-
matched CI is [-0.67732, +0.02481] pp. However that ranking fails the per-site/
seed easy ceiling. This is a useful protection/accuracy tradeoff, not evidence
that forest ranking dominates. Net-gain matching does not exhaust alternative
neural risk-ratio rankings or establish a matched displacement budget.

| Primary, seed-averaged site | ADE gain | Easy degradation |
| --- | ---: | ---: |
| coupa | 4.00672% | -4.97775% |
| deathCircle | 5.60814% | -0.19124% |
| gates | 1.96777% | -0.90186% |
| hyang | 2.53852% | -0.92418% |

The empirical positive-error easy and complete zero-reference checks pass.
The registered **primary superiority gate fails**, so the conjunction fails.
Better FDE, fewer harmed observed cases or a secondary point score cannot
replace the registered primary criterion after readout.

## Conditional Risk and Selection

On the ramp forest's selected complete-label rows, mean harm is overestimated
in all 12 held site/seed views: realized/predicted ratio median **0.6093**, range
[0.4543, 0.9551]. On the neural head's own choices it underestimates harm in
all 12, ratio median **2.7861**. Those are different selected populations.

The fixed-region cross-check makes the distinction explicit. On the neural
choice set, the forest underestimates in 6/12, with median ratio **1.0231**.
On the forest choice set, the neural head underestimates in only 1/12, median
**0.5855**. Both score estimates and which rows are admitted matter. The forest
has not been shown calibrated on all regions; its own selected mean is visibly
conservative. These group means do not guarantee any individual outcome.

The fixed fit/held audit also finds this conservatism in fitting groups: the
ramp forest's realized/predicted selected-harm ratio median is **0.3074** in
36 fitting-site groups versus **0.6093** in 12 held groups, with no mean
underprediction in either set. Thus a fit-to-held shift remains; a larger
conservative margin absorbs it in these observed group means. Do not call
that accurate calibration or proof that distribution shift has disappeared.
Neural fitting/held diagnostics exactly recover the preceding audit.

The same features contain useful protective information under this standard
estimator, so the earlier failure cannot simply be described as "no signal in
the features." But this comparison does not isolate the estimator, objective,
regularization or selection effect, nor prove sufficient intent information for
every scene. Do not change a held-case-specific rule or call this tree method
a novel multimodal world model.

## Missing Labels and Interaction

Primary switching falls from 33,793 to 22,539 repeated seed instances, a rate
of **4.2747%** of all indexed query/seed instances. Of those, **2,663** have an
incomplete future, including **190** with no ADE labels. They are retained and
not counted as harmless. The deterministic full-grid absolute gain lower bound
is positive in every site/seed, even assigning unknown labels their maximum
triangle-inequality harm. The smallest is **0.02584 annotation pixels** in
gates seed43. This supports aggregate 12-step gain under the stated missing-label
bound, not per-query safety or easy-group protection for unknown outcomes.

Past-context excess-proximity totals are lower than both neural strict and
equal-count neural ranking in each seed: forest **11.9906, 11.1304, 11.1077**;
neural strict **18.4518, 17.7104, 13.7444**. Each seed has 20,932 scene queries,
265,016 supported edges, 7,299 unsupported edges and 5,748 unsupported context
occurrences. Discrete second-difference means are also lower in each scene
than both controls. These are native-pixel proxies, not physical acceleration,
collision guarantees or evidence of a newly trained joint-agent mechanism.

## Verification and Next Decision

All 24 endpoints replay exactly over 1,054,536 cost-score rows. Separate code
checks fitting labels/weights over 3,163,608 reused fitting-arm row instances,
explicit sums of tree predictions, 72 fixed choices, 576 scene reductions,
unknown-label bounds and gate arithmetic. Context, smoothness and conditional
summary calculations are replayed, not independently reimplemented. Twenty-four
scoped tests plus two verifier tests pass; the full legacy suite was not run.
Shared data and the same implementing agent do not constitute independent
research confirmation.

Retain the forest as a **protected development comparator**, not a new deployment
or a claimed neural/world-model contribution. The neural risk head currently
has no demonstrated advantage over this simpler conventional estimator under
the protection requirement. Before more head training, distinguish risk ranking
from conservative score levels with a separately fixed, same-count comparison;
do not select new thresholds on these outcomes. Independent scene admission and
calibration/confirmation roles remain essential and unresolved. Four explored
sites and three seeds do not supply that evidence. The project is **not yet a
CVPR submission candidate**; the pinned manuscript is unchanged.

See [registered design](registration.md), [full metrics](analysis.json),
[replay](replay.json), [separate checks](separate_verification.json), and
[execution instructions](execution_notes.md). The additional
[fit/held diagnosis](fit_held_diagnosis.json) retains all fitting and held groups.
