# Incremental Joint Control: All Registered Results

Frozen forecasters and cost heads; no new training. Restricted past-hash query population, not full parent rows.
36 dependent role/seed/event views; four source localities per view,12 unique localities. No independent confirmation.

| Policy | All ADE gain vs incumbent | Positive / negative CI | Hard ADE gain | Worst easy degradation vs CV |
|---|---:|---:|---:|---:|
|old_stop|+0.000000% to +0.000000%|0 / 0|+0.000000% to +0.000000%|1.2150635242035523|
|full_add|+0.029490% to +1.086160%|29 / 0|-0.022611% to +0.643400%|1.394770753218677|
|half_independent|-0.000671% to +0.516968%|18 / 0|-0.010866% to +0.503421%|2.254973066465915|
|half_hash|+0.000000% to +0.366140%|19 / 0|-0.030371% to +0.289628%|2.254973066465915|
|half_unary|-0.001799% to +0.516610%|18 / 0|-0.010866% to +0.501238%|2.254973066465915|
|half_joint|-0.001799% to +0.516079%|18 / 0|-0.010866% to +0.503421%|2.254973066465915|
|scene_uniform|+0.000000% to +0.000000%|0 / 0|+0.000000% to +0.000000%|1.2150635242035523|

| Joint vs matched control | All ADE gain | Positive / negative CI | Hard gain | Hard positive / negative CI |
|---|---:|---:|---:|---:|
|half_independent|-0.012203% to +0.003400%|0 / 0|-0.002024% to +0.000000%|0 / 0|
|half_hash|-0.032315% to +0.297163%|10 / 1|-0.009842% to +0.406246%|5 / 0|
|half_unary|-0.000553% to +0.001953%|0 / 0|+0.000000% to +0.003175%|0 / 0|

## Identification And Safety Limits

There are13824 repeated query/views, but only1152 unique current queries.
73 query/views permit non-additive interactions; joint and unary differ in6 query/views (3 unique queries).
1 query/views failed a solver certificate; all matched variants then retain the independent reference, never an unmatched-rate advantage.
All future-unknown indexed rows are retained for inference; label-unknown costs remain undefined. Partial ADE and complete-window ADE are separate; FDE requires the actual endpoint.
Proxy reduction follows an optimized proximity objective and is not evidence of collision avoidance. Predicted harm budgets are not calibrated risk bounds.
Hash priorities are outcome-independent, but feasibility still uses the same predicted harm cap. Uniform control is not rate matched.
Unknown/non-indexed agents are omitted from the pair graph, not assumed collision-free. Scene selection only coordinates indexed eight-history agents.
Source-bootstrap intervals use3000 resamples and four localities; no IID-window or multiplicity-corrected claims. All adverse branches remain in groups/*.json.
Image-pixel raw-frame obs8/pred12, released detector tracks; not metric/seconds/human-gold/true3D/foundation/physical-safety evidence. No Stage5C, SMC or deployment.
