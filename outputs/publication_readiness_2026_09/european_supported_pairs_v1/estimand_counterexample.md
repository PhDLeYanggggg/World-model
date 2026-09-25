# Why More Ranking Pairs Need Not Repair the Risk Target

Status: posthoc constructed algebraic diagnostic. It uses no real-data outcomes,
does not fit a model and does not change the registered experiment. It establishes
an estimand mismatch is possible, not that this alone caused the empirical failure.

Let B be baseline error and H be positive harm. The controller intends to use
conditional moment risk E[H|x]/E[B|x]. The current pair auxiliary instead orders
realized q=H/(B+H), with weight abs(q_i-q_j). These orderings need not agree.

Consider two causal states. In state A, the two equally likely (B,H) outcomes
are (1,1) and (100,0). State B always has (B,H)=(1,0.03). Then:

| Quantity | State A | State B |
|---|---:|---:|
| Conditional mean baseline error | 50.5 | 1 |
| Conditional mean positive harm | 0.5 | 0.03 |
| Intended mean-harm / mean-baseline risk | 0.009901 | 0.030000 |
| Mean realized share E[H/(B+H)] | 0.250000 | 0.029126 |

At a 2% expected-cost budget, A meets the budget and B does not. Yet the mean
realized share orders A as riskier. For a complete balanced A/B pair set, the
weighted logistic objective is C+ softplus(-s)+C- softplus(s), up to a positive
constant. Here C+=(0.5-3/103)/2 and C-=(3/103)/2. Its optimum s=log(C+/C-) is
positive: the auxiliary favors the opposite order from the conditional moments.

The executable counterexample uses one locality with cyclic rows A1,B,A2,B.
There are no undefined labels, so old and supported-first losses are exactly
equal. The actual frozen auxiliary gives lower loss after underpredicting the
unsafe state's harm as 0.005 rather than its true conditional value 0.03.
The original moment and occurrence/severity losses still oppose such a change;
this calculation does not establish the optimum of their full combined objective.

## A Different Target to Test, Not a Proven Repair

For independent conditional observations, the signed cross-moment pair label
H_i B_j-H_j B_i has conditional expectation
E[H|x_i]E[B|x_j]-E[H|x_j]E[B|x_i]. With positive expected denominators, its sign
matches the intended risk order. In this construction its mean is -1.015, and
its idealized weighted-logistic optimum is negative, agreeing with the desired
order. This is elementary algebra, not a novelty or calibration theorem.

Real training pairs are dependent and batch weight normalization is itself
random; cross-product weights may also have high variance. The sign identity
does not prove finite-batch consistency, generalization, tail control or physical
safety. A later, separately registered fitting-only experiment could change the
pair target while retaining matched moments, capacity, risk limits and controls.
No such model has been trained in this experiment. Do not tune a coefficient or
threshold using the current development readout.

Run `.venv-pytorch/bin/python scripts/diagnose_m3w_ratio_ranking_estimand.py`.
Exact fractions and the current Torch loss produce
[the numerical receipt](estimand_counterexample.json). This is diagnostic-only,
not a new dataset, forecasting score, deployment result or independent test.
