# Risk Ranking Protects Easy Cases, but the Neural Head Loses More Utility

## Evidence Status

2026-09-22. **fresh_run:** fixed-count decisions, outcome readout, exact replay
and separate arithmetic verification. **cached_verified:** existing ramp
forecasts, tree/neural cost estimates and the complete source-exclusion chain.
**not_run:** new model fitting, independent calibration/confirmation, external
confirmation and raw-frame t+50. Registration/code were pushed as `259b2144`
before this new readout; prior development outcomes were already exposed.

This is SDD eight-observed/twelve-predicted annotation steps, raw stride12,
annotation pixels. There are 175,756 indexed queries in 33 recordings and only
four physical sites, all already used in design, with three existing seeds.
It is not an independent test, metric/seconds, true3D or foundation result.
Stage5C and SMC are off. No policy is promoted or deployed.

## Fixed Comparison

Keep the same ramp neural trajectories and the forest's existing per-site/seed
switch count. Compare forest versus neural scores and net gain G-H versus
relative harm H/(G+H). The latter has the same ordering as H/G for G>0; zero
benefit is assigned maximal risk, with no epsilon. Support uses only positive
forecast disagreement and a nonzero final observed step. Tie-breaking uses row
id. Future targets and masks cannot change choices or counts.

All four ranking policies make **22,539** repeated query/seed switches, **4.2747%**
of 527,268 query/seed instances. Forest ratio exactly reproduces forest strict;
neural gain exactly reproduces the previous count-matched control. The neural
ratio set is a subset of its 33,793 original strict choices, dropping 11,254.
These are offline allocation controls using full per-site score ranks, not
newly calibrated streaming deployment rules.

| Fixed policy | ADE gain vs CV | FDE gain vs CV | Hard gain vs CV | Aggregate easy degradation | Worst site/seed easy degradation | Complete exact-CV harmed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Forest ratio = forest strict | 3.53029% | 5.16628% | 3.76414% | -1.74876% | -0.13424% | 0 |
| Neural ratio at forest count | 2.80697% | 4.12167% | 2.97143% | -1.50838% | 0.16376% | 0 |
| Forest net gain at forest count | 4.07418% | 5.61974% | 6.17470% | 1.50834% | 2.70229% | 0 |
| Neural net gain at forest count | 3.82776% | 5.28068% | 5.80729% | 1.51097% | 2.55829% | 0 |
| Original neural strict, unmatched count | 3.39756% | 4.90029% | 3.67549% | -1.24935% | 0.91149% | 1 |

Gains average physical sites equally after averaging seed errors. Negative easy
degradation is improvement. Hard/easy retain fitting-only CV quantiles; exact-zero
CV remains a separate absolute-harm check, without epsilon. Counts and sums
include repeated correlated windows and seeds, not independent agents/events.

## Diagnostic Contrasts

| Fixed contrast | ADE gain difference | Paired physical-site bootstrap95% interval |
| --- | ---: | --- |
| Forest ratio minus neural ratio | **+0.72331 pp** | **[+0.61059, +0.82728] pp** |
| Neural ratio minus neural net gain | -1.02078 pp | [-1.44508, -0.67908] pp |
| Forest ratio minus forest net gain | -0.54390 pp | [-0.95069, -0.22822] pp |

All intervals use 3,000 paired physical-site resamples and seed38113. They are
conditional development diagnostics, not independent confirmation or
multiple-comparison-adjusted population claims. Three seeds are reused existing
fits, not additional fresh training. The forest ratio exceeds neural ratio in
all12 site/seed views; the smallest difference is +0.47964 pp.

| Site | Forest ratio ADE gain | Neural ratio ADE gain | Difference |
| --- | ---: | ---: | ---: |
| coupa | 4.00672% | 3.14783% | +0.85889 pp |
| deathCircle | 5.60814% | 4.91833% | +0.68981 pp |
| gates | 1.96777% | 1.41888% | +0.54889 pp |
| hyang | 2.53852% | 1.74285% | +0.79567 pp |

Forest ratio ADE gains by seed17/29/43 are 3.64813/3.45047/3.49226%; neural
ratio gives 2.95665/2.67442/2.78985%. Per-scene native errors, p95/p99 tails,
FDE and all masks remain in `analysis.json`.

**The previous primary forest-versus-original-neural superiority gate remains
failed:** its +0.13273 pp CI still crosses zero. This new conditional count
comparison does not replace that registered criterion or create a deployment
gate after seeing results. Neither net-gain policy satisfies every scene/seed's
2% easy limit, despite their greater average accuracy and hard gain.

## What the Control Explains

1. **Protection is not exclusive to the forest.** Neural relative-risk ranking
   at the lower count also protects observed positive-easy and exact-CV cases.
   Thus simply changing which neural scores are prioritized can repair those
   empirical checks without fitting another network, but it loses useful gain.
2. **Fewer switches alone is not the complete explanation.** At identical counts,
   forest ratio retains more accuracy in every site/seed. The forest and neural
   risk rules share14,120 switches and each has8,419 distinct choices.
3. **Raw net gain favors higher-impact forecasts and harms easy cases.** Within
   the neural head, risk/gain rules share only6,265 choices. Among the16,274
   ratio-only versus16,274 gain-only choices, complete-label harm sums are
   3,868.76 versus42,038.83 annotation-pixel error units. Observed easy gain
   sums are +6,506.19 versus -3,415.54. These sums explain selection turnover;
   they are not scene-balanced performance estimates or independent outcomes.
4. **Calibration remains unproven.** On its own risk choices, the neural head
   underpredicts complete-label mean harm in12/12 views, with median realized/
   predicted ratio3.84045. Forest own-risk choices have ratio0.60927 and0/12
   underprediction. On neural-risk choices the forest still underpredicts4/12;
   on forest-risk choices neural underpredicts1/12. Scores and accepted
   populations both matter. Passing observed easy checks is not risk calibration.

For the8,419 forest-only versus8,419 neural-ratio-only choices, complete benefit
sums are94,772.75 versus40,736.01 and complete harm4,605.15 versus11,649.24.
Complete supports differ (6,986 versus6,626), so missingness cannot be ignored.
The result supports useful ranking differences on these explored data, not
that every forest prediction is safer or that a new architecture caused the gain.

A common positive multiplier on predicted harm cannot change relative-risk
rank at fixed count. This control removes that particular conservative-level
explanation, but not row-dependent conservatism, training loss, regularization,
capacity or optimizer differences. Forest uses weighted fraction squared loss;
the current neural head uses weighted compositional log loss. The experiment
does not isolate architecture, and the forest is a standard strong comparator.

## Missingness, Displacement and Interaction

| Same-count policy | Selected incomplete | Selected unknown ADE | Minimum site/seed full-grid absolute-gain lower bound |
| --- | ---: | ---: | ---: |
| Forest ratio | 2,663 | 190 | +0.02584 pixels |
| Neural ratio | 3,023 | 353 | -0.08235 pixels |
| Forest gain | 5,491 | 585 | -0.19266 pixels |
| Neural gain | 5,779 | 684 | -0.26171 pixels |

Unknown outcomes are retained, not called harmless. Full12step triangle bounds
are positive at every site/seed only for forest ratio among these four controls.
Neural ratio's lower bound is negative in gates for all three seeds. These are
deterministic missing-label bounds on aggregate gain, not confidence intervals,
individual safety or protection for unknown easy labels.

Matched count does not match displacement mass. Total causal disagreement on
selected repeated instances is502,232.15 forest ratio versus440,867.75 neural
ratio, and846,211.84/818,913.42 for forest/neural net gain. Forest can take
larger-impact changes at the same count; no equal-displacement claim is made.

Past-context excess-proximity totals by seed17/29/43 are11.9906/11.1304/11.1077
for forest ratio and11.9237/11.7601/11.1111 for neural ratio: no uniform dominance
over that neural control. Both are lower than their own net-gain versions.
Scene-average second-difference smoothness is also mixed between the two ratio
rules. These are annotation-coordinate proxies, not real collisions, metric
acceleration, a safety certificate or a newly learned joint-agent mechanism.

## Verification and Research Decision

All72 policy/view choices replay exactly. Separate code sorts direct H/G instead
of the implementation's normalized fraction, reconstructs candidate/error
arithmetic, checks576 scene reductions, bounds/overlaps and96 selected-harm
groups. Context, smoothness and turnover records are replayed, not separately
reimplemented. 22 scoped tests pass, including14 new ranking/verifier tests;
the full legacy suite was not run. This is same-agent engineering verification,
not independent research replication.

Retain the forest as a necessary conventional development reference and keep
deployment unchanged. Do not spend another general width/epoch sweep or
threshold search. A narrowly controlled neural comparison can next match the
forest's normalized cost target and fixed squared-loss weighting on the same
ramp action, to distinguish the remaining loss-versus-estimator confound. The
older bounded-fraction trial used another action, weighting and budget, so it
does not already answer that matched question. Register it before fitting.

Independent scene admission, calibration/confirmation purposes and the risk
functional still require the outstanding scientific decisions. Do not reuse an
exposed producer chain as independent calibration. Joint-agent contribution
also remains absent: the previous joint-control experiment was null, and a new
pairwise-weight sweep on these results would not repair it. New independent
support and a training-supported interaction mechanism remain paper-level gaps.
CREATE access/project/queue are unverified; this diagnostic needed no remote job.

**Not yet a CVPR submission candidate.** The pinned manuscript and failed
historical comparisons remain unchanged. Historical Stage26/37 scores remain
exploratory, not recertified by this study. See [registration](registration.md),
[full metrics](analysis.json), [replay](replay.json),
[separate verification](separate_verification.json) and [commands](execution_notes.md).
