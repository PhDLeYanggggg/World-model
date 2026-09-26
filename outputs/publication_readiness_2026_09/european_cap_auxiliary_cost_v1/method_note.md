# Method and Interpretation Notes

## What Is Estimated
Let X denote the causal input vector and d(X) the known disagreement envelope
between the two frozen forecasts. Let Y_H be realized nonnegative excess
forecasting cost and Y_E its positive-CV-easy component. The target interface
enforces0 <= Y_E <= Y_H <= d(X). The learned costs satisfy the same nesting:

    z = SiLU(W X + b)
    H_hat = d(X) sigmoid(a(z))
    E_hat = H_hat sigmoid(e(z))
    p_cap = sigmoid(c(z))

The third output is an auxiliary event probability. It never scales E_hat.
Frozen reference-cost components are copied unchanged. Calling this a nested
bound describes the output parameterization, not a calibrated risk guarantee.
In particular, a bound relative to forecast disagreement is not physical safety.

## Objective and Controls
For fitting-locality-excluded teacher H_inner, the auxiliary label is
I[Y_E > H_inner]. Common-meta easy labels use only the three fitting
localities. Future outcomes serve only as labels. No already fitted cap-event
classifier prediction enters X. At held evaluation the event is relative to
the corresponding outer teacher, which has three fitting localities rather
than two; that producer-size transport is unresolved.

The cost objective averages squared H/E errors after fixed fitting RMS
normalization. Three arms add0 times binary log loss,1 times true-event log
loss, or1 times within-locality-shuffled-event log loss. The randomized label
control preserves each fitting locality's prevalence and missingness. It is
a fixed negative control, not a randomization significance test.

Initialization, parameters, optimizer, batches and updates are matched.
All event-head parameters exist in all arms. A zero event-loss coefficient
leaves cost gradients independent of the event label, as tested explicitly.
Raw total objectives are not directly comparable across the cost-only and
auxiliary arms; report cost loss and true-event log loss separately.

## Why Event Learnability May Not Improve Costs
Event occurrence and expected magnitude are different functionals. A model
can rank cap exceedance and still misestimate how large a forecast error is.
An accurate conditional mean can also lie below some realized outcomes;
those outcomes do not alone prove downward conditional bias. Auxiliary loss
can help or interfere with a finite shared representation. It supplies no
general theorem of lower held cost MSE. That is the experiment's question.

The linear classifier's previous success does not prove that a shared
nonlinear cost encoder benefits from the same task. Nor would an improvement
over the new cost-only arm alone prove a gain over the original strong head.
The separate shuffled-label comparison tests whether the true task carries
useful information beyond adding an auxiliary optimization objective.

## Evaluation and Limits
Keep every assignment and seed. The primary measure is H_E MSE improvement
on positive-disagreement rows, with all-H error, top-tail capture and mean
coverage-ratio guards. No negative guard interval is a predeclared screening
condition, not a formal noninferiority certificate. Four-locality exploratory
intervals are limited; overlap is not equality. Repeated windows, seeds and
assignments are not extra independent localities.

Full and motion-only families change both forecast pair and target population.
Their contrast is not a retrained causal feature ablation. Historical source
development exposure remains even when a new model excludes one locality
during fitting. Independent selection, calibration and confirmation remain
separate and unopened. No new policy is evaluated in this study.

Obs8/pred12 are native annotation steps; coordinates are detector-image pixels.
No metric, time-in-seconds, human-gold, true-3D, foundation-model or physical
safety interpretation is licensed. Stage5C and SMC remain off.
