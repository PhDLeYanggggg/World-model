# Symmetric Utility: Better Cost Estimates, No Neural Advantage

## Verdict

The single-factor repair improves expected-cost estimation but does not rescue
the neural-trajectory hypothesis. All 24 full-pointwise neural-versus-protected
damping contrasts remain negative, with negative conditional locality intervals.
All 24 hard and 24 joint-pilot contrasts also favor protected damping.
No deployment or reserved-data decision changes. Not yet submission-ready.

## What Was Actually Run

Fresh training of 18 Torch utility heads: two candidates, three source folds,
three seeds, 2,000 updates each, totaling 36,000. The 100-update pilot resumed
inside this budget. All 18 completed before comparative outcome readout.
All 72 risk heads and 18 old utility heads are cached_verified. No new
trajectory/JEPA training occurred. All 48 fixed policy views are retained.

Both candidates changed only utility loss, from underharm4 to symmetric MSE.
Each new head has exactly the old head's preprocessing, initialization
constants, cost scale, fitting support, draw counts and sampler RNG. No unknown
future-label row entered supervision. Forecasts, risk outputs, 2% predicted-risk
budget and support guards stayed fixed. This is not threshold relaxation.

## Fixed Easy-Event Neural-Risk Views

These views continue the preceding report's fixed examples; they are not winners
selected from this experiment. Percentages below are ADE improvement over CV.

| Seed | Old neural utility | New neural utility, conditional 95% CI | New protected damping, conditional 95% CI | Neural worst-locality easy degradation | Neural switch rate |
|---|---:|---|---|---:|---:|
| 17 | 0.2382% | 0.2719% [0.1513%, 0.4029%] | 2.0901% [1.0720%, 3.0996%] | 0.0496% | 2.2661% |
| 29 | 0.1663% | 0.1982% [0.0869%, 0.3238%] | 2.1974% [1.1799%, 3.2154%] | 0.0000% | 0.4411% |
| 43 | 0.4310% | 0.4188% [0.1553%, 0.7174%] | 2.1559% [1.1706%, 3.1354%] | 0.6885% | 0.6427% |

All three listed neural and damping views harm none of the four observed
zero-CV cases. The three damping views improve easy ADE in every locality.
Direct new-versus-old neural policy gains are 0.03386%, 0.03193%, -0.01270%;
the first two conditional intervals exclude zero, the third includes zero.
Corresponding damping gains are 0.28849%, 0.26685%, 0.26658%, all with positive
conditional intervals. The repair benefits the simple motion control more.

Across all views, 18/24 neural changes are positive and 10 have positive
conditional intervals. Damping is positive in 21/24, with 15 positive intervals.
Neither candidate has a new-versus-old contrast with a strictly negative
interval. This does not establish consistent improvement in every seed/view.
The direct neural-versus-damping point estimates range from -2.0768% to -0.2228%.

## What the Mechanism Check Supports

On equal-weight held source localities, neural utility-harm bias falls from
4.4745/4.6216/4.5779 pixels to 0.1051/0.3505/0.1960 pixels. Harm MAE falls from
7.3469/7.6879/7.5110 to 4.3705/4.6698/4.3766 pixels. These are expected-cost
regression diagnostics, not calibrated probability or certified risk.

The locality-averaged fraction with positive predicted utility rises from
25.56/15.76/18.19% to 69.34/55.07/63.81%. That statistic uses ADE-supported rows,
equal locality weighting; it is not directly comparable to the full-row switch
rate's denominator. Nevertheless, the fixed risk controller still permits few
neural interventions. Utility bias was real, but correcting it alone is not
sufficient to establish reliable neural intervention or superior dynamics.

## Safety and Joint Negative Results

Full-pointwise observed safety passes 22/24 neural and 21/24 damping views,
unchanged in count from the old utility experiment. Neural easy/ridge/no-guard
harms one zero-CV row in seed17 and seed29. Damping easy/ridge/no-guard exceeds
the worst-locality easy limit in all three seeds: 2.4901/2.7435/2.7872%.
These failures are retained, not hidden behind the fixed example table.

Joint observed easy preservation passes 12/24 neural and 20/24 damping views.
No defined matched-count joint-versus-unary interval is strictly positive.
Fourteen neural and twelve damping matched-count contrasts are undefined
because the required supported localities are missing; no locality is dropped.
The joint population has no zero-CV cases and cannot validate their protection.

One exact-count neural joint call returned `solver_solution_invalid_floor`:
seed43, fold2, all-event ridge, no guard; recording98/frame17090, 21 agents.
Its safe fallback is retained and the query is not treated as matched/optimal.
The saved receipt does not establish whether timeout or numerical feasibility
was the underlying cause. It was not selectively rerun with a looser budget.
All retained decisions satisfy the frozen predicted-risk constraint.

## Verification and Scope

Full metrics reconstructed exactly; all 18 checkpoints reproduce 4,096 sampled
rows each exactly. Independent accounting reconstructs all 144 full-pointwise
decision receipts and checks query counts, frozen risk artifacts, risk budgets,
source support and matched intervention counts. All 179 scoped tests pass
across 23 files; this is not the full legacy suite.

318,969 targets from 12 opened source localities; 311,922 ADE-supported and
7,047 unknown; joint pilot 1,152 queries / 6,116 targets. Three seeds and 3,000
source-locality bootstrap resamples provide conditional development evidence,
not independent confirmation. Reserved roles remain closed. Released detector
tracks are not human gold. Image-pixel obs8/pred12, raw stride12, not raw t50,
seconds, metric, physical safety, true3D or foundation-model evidence.
Stage5C and SMC remain off.

## Next Scientific Decision

Separate source-only risk-estimation error from genuinely unpredictable neural
harm, using the now-frozen symmetric utility heads. Check the existing nested
producer shift and conditional event-head reliability before changing risk
objectives. Any new fit must be a predeclared matched ablation for both candidates,
not a risk-limit relaxation or a threshold selected from these outcomes.
Protected damping remains a mandatory competitor. If neural prediction cannot
beat it safely, the contribution must be narrowed rather than relabeled as a
successful neural world model. Independent calibration and confirmation are
still unrun, and joint coordination still lacks a positive controlled result.

[All results](results.md), [light metrics](summary_metrics.json),
[all-view figure](utility_ablation.svg), [training losses](training_losses.md),
[Chinese reproduction guide](operation_zh.md).
