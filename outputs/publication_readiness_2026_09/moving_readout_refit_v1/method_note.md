# Why Solving The Last Layer Does Not Solve Admission Calibration

## Algebraic Control

For a frozen causal design X, fixed empirical query weights W and signed
risk-excess target y, weighted least squares solves

    min_beta (X beta - y)' W (X beta - y).

At a full-rank optimum, X' W (X beta - y) = 0. The observed relative normal
residual is approximately 4e-15 to 1e-14 for this experiment. The actual
float32 deployment path also lowers the original fitting data loss.

That equation constrains correlations with columns of X, not with every
nonlinear indicator S(X) selected by the resulting policy. Unless the selected
indicator is represented in the tested span, it does not imply

    sum_i W_i S(X_i) (prediction_i - target_i) = 0.

This is an algebraic limitation, not a new theorem about world modeling.
The regression tests include a synthetic linear-optimum counterexample;
it is not counted as real-data evidence. The fitting loss uses normalized
targets and query weights, while reported excess sums use available outcomes
in native image units. Those are intentionally distinct measurements.

## Related Work And Boundary

Hebert-Johnson et al. study calibration over specified, potentially overlapping
computationally identifiable populations. Their prediction-dependent calibration
sets and adaptive generalization analysis make clear that marginal prediction
accuracy and subgroup calibration are different requirements.
[Original ICML 2018 paper](https://proceedings.mlr.press/v80/hebert-johnson18a.html),
[full text, Sections 1-2](https://proceedings.mlr.press/v80/hebert-johnson18a/hebert-johnson18a.pdf).

Application to M3W is an inference, not a transferred guarantee. Our outcomes
are continuous signed excesses with bounded unknown-label contributions,
overlapping trajectory windows, source shift and a separate reference-support
contract. The paper's probability-calibration guarantees cannot be asserted
for this experiment. No multicalibration algorithm was implemented or tested
here, and it would not resolve a genuinely absent easy-reference denominator.

## What To Change Next, Not Yet Run

Do not repeat an identical last-layer optimizer or claim this standard solve
as a new method. The next controlled model change should address the nonlinear
selected-set error using causal information, with fitting-source-only learning
and the same source roles and thresholds. Before training, distinguish it from
the already-tested online admission penalties, event-magnitude heads and
selected-set calibration studies. Any data-driven partition must be fitted
without held SOURCE or H outcomes. The current three-context result cannot
select a favorable context or threshold for deployment.

Original 2% risk, support, empty-control and signed easy-net rules remain fixed.
Old failures and the original unavailable primary remain unchanged. No new H,
metric/seconds/physical-safety, human-gold, true 3D or foundation claim.
Stage5C and SMC remain disabled.
