# Fixed-Count Pair-Aware Assignment

This is a standard structured-assignment control,not a new optimization theorem.
Its purpose is to separate budget pooling from explicit trajectory coupling.

For current-query target i,state0 denotes its frozen fallback,and states1..7
denote the original seven candidate forecasts. Let x(i,a) be a one-hot action.
Unavailable or nonpositive-predicted-gain interventions are excluded exactly
as in the original decoder. The fallback remains available to every target.

The original predicted benefit/harm/reference/easy-reference/easy-harm moments
are B,H,R,ER,EH. Fallback contributes zero intervention moments. We preserve:

```text
sum_a x(i,a) = 1                  for each target i
sum_i sum_(a>0) x(i,a) = original query intervention count
sum_i,a [H(i,a)  - 0.02 R(i,a)]  x(i,a) <= 0
sum_i,a [EH(i,a) - 0.02 ER(i,a)] x(i,a) <= 0
```

These constraints concern predictions,not realized error. The original full
recording/easy-risk evaluation is still required and cannot be replaced by
the number of feasible queries.

Restore floor/candidate forecasts to the common local image frame using only
their current origins. On the past-defined undirected four-nearest-target
graph,define C(i,j,a,b) as the requested-step mean of the positive decrease
in pair distance relative to both targets taking their fallbacks,divided by
the geometric mean of their frozen past scales. C(i,j,0,0)=0.

The objective is:

```text
- sum_i,a [B(i,a)-H(i,a)] x(i,a) / sum_i past_scale(i)
+ 1.0 / |edges| * sum_(i,j),a,b C(i,j,a,b) y(i,j,a,b)
```

The pair term vanishes if the graph is empty. The coefficient is fixed before
this experiment's outcomes and is not a learned physical conversion.

Continuous y>=0 variables have row marginals x(i,a) and column marginals
x(j,b). Since each x is binary one-hot,the only nonzero pair entry is the
chosen pair,with value1. Thus these transport constraints represent the exact
discrete pair objective for an integral solution. SciPy/HiGHS solves the MILP;
finite-budget nonoptimal returns are not accepted as proven optima.

The frozen parent is feasible with exactly the same intervention count. A
two-second solver failure or failed numerical count/budget/objective postcheck
returns that parent,not an empty selection. Queries without a pair opportunity
also retain the parent without a new optimality assertion. The cohort is
never filtered by future labels or solver difficulty.

Thirty-two small random problems,including changing selected membership at a
fixed count,are checked against exhaustive enumeration. Rigid/common-scale
geometry invariance,original complete-pair formula agreement,planned recovery
and numerical-failure fallback are checked separately. These are finite
engineering tests,not proof of predictive utility or risk calibration.

Only current query targets enter this graph;all-visible-agent consistency is
not established. Oblique detector-silver pixels do not support body-aware
collision,meters,seconds,or real-time sensor-causality claims. An improvement
in C would establish a geometric proxy change only;the original ADE completion
bounds and recording-risk endpoints decide scientific advancement.
