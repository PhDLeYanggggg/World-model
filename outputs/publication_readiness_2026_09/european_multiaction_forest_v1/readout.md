# Matched Conventional Cost Estimator: Verified Readout

Result source: fresh ExtraTrees training and inference; frozen neural forecasts,
source packets and static-scene neural controls are cached_verified. SOURCE
inner-held is exposed development, not independent confirmation or whole-pipeline OOF.

## What Was Actually Trained

All 54 registered forests / 6,912 primary trees complete (PID 8500, exit 0).
Training process wall time 2454.886983s; sum of primary fit times 2309.918388s;
peak RSS 10,744,889,344B; all private forest checkpoints 335,413,378B.
No new trajectory forecaster or Torch optimizer updates. The 128-tree refit and
all 144 inference groups replay exactly. Three controller seeds are not three
independently trained forecasting pipelines.

Both estimator families receive the same 398 causal/static fields, seven action
identities and causal envelope information, fitting-only normalization and the
original query-draw multiplicities. Standard forest quadratic regression with
post-fit projection is not the neural underharm/ranking loss or decoder; this
is an estimator/objective control, not a matched-FLOP one-factor ablation.

Inherited calibration MSE retains the parent all-seven-action definition,
including inactive-bank predictions. It is not active-bank policy error;
the separate fitting-region mechanism check uses legal bank actions.

## Primary Fixed Comparisons

| Bank | Forest known ADE gain vs floor (%) | Static neural gain (%) | Forest minus neural (pp), 95% CI | Completion-lower contrast (pp), 95% CI | Full record-risk support |
|---|---:|---:|---|---|---|
| physical_only | -0.000083 | +1.076598 | -1.076681, [-2.523074, -0.258453] | -2.923766, [-7.834809, -0.350935] | 120/186 |
| physical_neural | -0.000073 | +0.982769 | -0.982843, [-2.301874, -0.242325] | -3.200105, [-8.549639, -0.355064] | 118/186 |
| neural_only | +0.044378 | +0.433381 | -0.389003, [-0.666399, -0.152410] | -1.812941, [-4.287457, -0.257930] | 65/186 |

Within forests, mixed minus physical known ADE: +0.000009pp, CI [-0.000024, +0.000052].
Its completion-lower contrast: -0.000005pp, CI [-0.000047, +0.000042].
Intervals use 3,000 locality resamples after averaging repeated context/controller
seed views; these are conditional development intervals. Count-matched contrasts
are utility-only. They do not repair recording-risk failures or undefined support.

Completion contrasts divide worst-case net utility differences by the fixed
known floor-error mass. They are not error-reduction percentages for a fully
observed future population with an unknown denominator.

## Known-Outcome Hard/Easy Supplement

| Subset | Repeated known row views | Empty required views | Forest mixed ADE gain (%) | Neural mixed ADE gain (%) |
|---|---:|---:|---:|---:|
| hard | 100056 | 6 | +0.000001 | +0.639368 |
| easy | 93351 | 0 | -0.002421 | +1.298352 |

These ratios pool known error mass over repeated views, not independent samples
or the primary locality macro. Empty required hard views remain not_estimable.
Hard/easy cuts were fixed on fitting data. Unknown outcomes are not imputed.

## Fitting-Only Cost Mechanism

All 1,332 paired recording/head/bank views replay exactly, with no updates.
Both estimators see identical observed cost mass within each fixed region.

| Mixed-bank fitting region | Estimator | Known/unknown action views | Median predicted/observed H | Median predicted/observed EH | EH finite/undefined record views |
|---|---|---:|---:|---:|---:|
| all_legal | forest | 5256810/113040 | +0.963474 | +1.016480 | 444/0 |
| all_legal | neural | 5256810/113040 | +1.498049 | +1.026498 | 444/0 |
| frozen_neural_selection | forest | 69708/853 | +6.882285 | +4.576410 | 341/103 |
| frozen_neural_selection | neural | 69708/853 | +2.241466 | +0.021367 | 341/103 |
| frozen_forest_selection | forest | 6946/80 | +4.108277 | +4.456405 | 323/121 |
| frozen_forest_selection | neural | 6946/80 | +2.496244 | +0.143958 | 323/121 |

Medians condition on a positive observed denominator; undefined counts remain.
All-legal candidate pools are not deployment policies. Fixed-region differences
separate changed cost estimates from selection composition, but cannot uniquely
identify architecture as the cause: estimator objectives and decoders differ.

## Safety And Interpretation

Mixed forest: 3,457 selected row views, 43 unknown selected; 0 known all-risk violations, 2 known easy-risk violations, 66 undefined easy recording views.

Mixed forest neural-action views: 182; neural-only forest neural-action views: 31959.

Full required recording all/easy 2% contract supported: False.
Forest beats static neural on known paired CI: False.
Forest completion-lower paired CI positive: False.
Neural candidate increment completion CI positive: False.
Do not replace an unknown-completion failure with a known-only win. No policy,
seed, checkpoint or 2x/4x sensitivity was chosen after this readout. There is
no deployment promotion, independent confirmation, or submission-ready claim.

## Scope And Next Scientific Decision

EuropeanSquares SOURCE uses oblique webcam detector-track box midpoint pixels,
obs8/pred12/stride12 raw frames. It is not SDD human-gold, metric/seconds,
true 3D or a foundation model. No future endpoints, central velocity, test goals,
test normalization or future-support masks enter inference. Independent
selection/calibration/confirmation and DroneCrowd remain closed. No Stage5C or SMC.

Use this result to decide whether cost-estimator capacity is the limiting factor.
If a stronger conventional estimator still fails the full recording contract,
the next experiment must address identifiable conditional risk or support, not
repeat another neural architecture/threshold sweep on this exposed readout.
