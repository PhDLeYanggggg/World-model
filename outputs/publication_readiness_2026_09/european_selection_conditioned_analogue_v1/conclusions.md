# Matching the Reference Population Helps, but Does Not Repair Risk

## Material Passport

Result source: **fresh_run** for 72 fitting-only comparisons and event/magnitude
accounting; **cached_verified** for inherited models, frozen query selection and
the five full-bank controls. New neural training, policy execution, held readout
and independent evaluation are **not_run**. No model is promoted.

## What Changed

Only supervised donor eligibility changed: donor labels now come from rows
selected by the same frozen original policy and head seed. The original queries,
source-qualified recording exclusion, full-bank feature normalization, 380 causal
features, k32, clipping and source-normalized evaluation remain unchanged.
All five inherited controls exactly match their previous predictions and scores.
Selected-label permutations use the smaller donor sets, not identical old shuffles.

Run PID78105 exited0 after231.309565s, with6,710,427,648B peak RSS. There were
zero neural optimizer updates. All72 views and62,902 query occurrences remain
supported:61,335 known and1,567 unknown. Selected donor counts after recording
exclusion range393 to15,117, so the improvement is not caused by dropping
unsupported queries. These are repeated occurrences, not independent people
or recordings. No unknown outcome is assigned a safe label.

## Full-Roster Results

MSE means source-normalized signed-risk error averaged equally over72 views,
not trajectory ADE/FDE. Harm masses sum normalized known-outcome occurrences.

| Estimator | All risk MSE | Easy risk MSE | Predicted easy harm |
|---|---:|---:|---:|
| Frozen original |0.079458745|0.000309373|1.375607|
| Estimation continuation |0.078065818|0.000309427|2.268195|
| Full-bank mean |0.174918744|0.005576636|1817.041412|
| Full-bank nearest32 |0.297406867|0.004740689|697.167173|
| Full-bank shuffled32 |0.216899278|0.011333992|1836.233479|
| Selected-bank mean |0.091997194|0.001709022|946.449805|
| Selected-bank nearest32 |0.101728186|0.002365719|370.305280|
| Selected-bank shuffled32 |0.101788361|0.002384333|959.928872|

Conditioning nearest donors on the original policy reduces easy-risk MSE by
**50.10%** and all-risk MSE by **65.79%** relative to full-bank nearest. All three
head-seed averages improve against that comparator. This isolates an effect of
the reference-population construction under the fixed experiment, not the sole
cause of risk failure or an independent generalization result.

The magnitude screen still fails: easy-risk MSE is **7.65x original**, and
**0/18** context/head-seed pairs beat both original and full-bank nearest.
Selected nearest is also38.43% worse than selected mean, and only0.78% better
than selected shuffled on easy MSE. Matching histories adds little to this
particular conditioned-bank MSE contrast. Do not replace the registered nearest
arm with the more favorable mean after seeing results; mean also fails to beat
the original head and badly overpredicts aggregate harm.

## Event Frequency Is Not Enough

Here an easy event is **positive harm on an easy sample**, not easy membership
alone. Neighbors estimate its probability and positive H/envelope severity;
their product reconstructs the predicted mean harm fraction. Query labels are
used only for the following diagnostic scoring, never as inputs.

| Known-cohort diagnostic | Full-bank nearest | Selected-bank nearest |
|---|---:|---:|
| Mean all-harm event probability |25.0147%|19.7416%|
| Observed all-harm prevalence |19.8027%|19.8027%|
| Mean easy-harm event probability |6.0929%|6.2746%|
| Observed easy-harm prevalence |6.1335%|6.1335%|
| Easy-harm event Brier |0.0581947|0.0571717|
| Easy-harm SSE on event-positive queries |39.001010|40.662720|
| Easy-harm SSE on non-event queries |82.241591|36.437111|

All-harm probability becomes close to aggregate prevalence. Easy-harm probability
was already close before the change. Yet selected nearest predicts370.305280
easy-harm mass against95.560642 observed, still **3.88x**. Of that predicted mass,
363.075059 occurs on realized non-events and7.230221 on event-positive queries.
The latter is below the95.560642 observed harm on positive events. Event-positive
harm SSE increases4.26%, while non-event SSE decreases55.70%. The reduction
therefore does not establish accurate localization or severity on actual events.

These event/non-event sums are diagnostic on realized outcomes. A probabilistic
forecast can properly assign nonzero probability to a non-event, especially for
rare events; the98.05% non-event predicted-mass share is **not** a false-positive
classification rate or standalone proof of miscalibration. Together with the
MSE, overpredicted total mass and underpredicted realized-event mass, it rejects
the claim that matching average event frequency alone has repaired risk.

346 of3,762 positive easy-harm query occurrences have no positive event among
their32 selected donors. Those cases are counted, not removed. Conditional
severity MSE uses only positive-query/positive-donor support and cannot establish
an unconditional improvement. Full-roster AUROC/AP remain undefined when any
view lacks event support; the26 easy non-event views are not silently excluded.

## Adverse Slices and Decision

Locality020 contributes about60.67% of the equal-view selected-nearest easy MSE.
It has355 repeated queries,342 known and18 easy-harm events, not355 independent
recordings. Its easy MSE improves versus full bank but remains0.017224676, far
above original0.000009287. Locality048 worsens from0.001335369 to0.003434859;
context single0/controller1 also worsens. All remain in the primary result.

Keep deployment, reserved data roles and2% positive selected harm/reference
unchanged. Do not fit a global scale to the reported mass ratio, retune k or
remove adverse sources. The next repair must address conditional cost allocation,
not only event prevalence. Reconcile the earlier cost-aligned Newton and hurdle
controls, then test a loss-aligned local cost fit on the same selected donor bank:
uniform H/envelope averaging need not minimize source-normalized signed-risk MSE.
That follow-up is **not_run**, and would not by itself certify a safe policy.

Upstream model, preprocessing and selection exposure remain; donor exclusion
is not whole-pipeline OOF. No confirmatory CI or p-value is attached to fitting
repeats or one shuffle seed. Protocol remains obs8/pred12 rawstride12,
detector-silver pixel/local; no metric/seconds/gold/physical-safety/true3D/
foundation claim. Stage5C/SMC remain off. Research and submission goals are open.
