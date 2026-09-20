# Separating Single-Agent Geometry From Joint Intervention

This is a necessary mechanism control, not a novel theorem or a demonstrated
prediction improvement. It applies to the existing binary choice between two
fixed causal rollouts. The primary metric and risk tolerances are unchanged.

## Decomposition

For one edge, with binary switches x_i,x_j and P(0,0)=0:

```text
P(x_i,x_j) = P(1,0)*x_i + P(0,1)*x_j
            + [P(1,1)-P(1,0)-P(0,1)]*x_i*x_j.

U(x) = -sum_i gain_i*x_i/n
       + lambda/m * sum_edges [P(1,0)*x_i + P(0,1)*x_j].

R(x) = lambda/m * sum_edges d_ij*x_i*x_j,
d_ij = P(1,1)-P(1,0)-P(0,1).
Full objective J(x) = U(x)+R(x).
```

Although P is nonnegative, d_ij may be negative. Clipping d_ij would change the
objective. No-edge or zero-weight problems have R=0. At exact count zero or one,
no product can be active. A nonzero edge alone is not evidence of useful
coordination: it may connect unsupported agents, or all feasible solutions may
have the same product value.

The proposed controls are:

1. Risk-only independent reference: minimize predicted negative net gain under
   original support, harm and maximum-count constraints. Its causal count is k.
2. Geometry-aware independent: minimize U with exactly k switches and the same
   support and original predicted-harm cap.
3. Full pairwise: minimize J on that identical exact-count feasible set.

The coherent risk estimate remains max(predicted_harm,-predicted_gain).
Single-agent geometry is an objective term, not a new estimate of forecast harm.
It must not alter the risk budget through an adjusted gain. The exact count is
determined before reading labels. Future validity cannot remove agents from
the graph or feasible set. Forecasts, input roles and common coordinate/time
restoration remain fixed. No forecaster is rerun by the opt-in adapter.

## A Useful Bound, Not A Safety Certificate

Let x_U minimize U and x_J minimize J on the same nonempty feasible set.
Then

```text
0 <= J(x_U)-J(x_J)
   <= R(x_U)-R(x_J)
   <= lambda/m * sum_edges |d_ij|.
```

The first inequality follows from full optimality; the second from
U(x_U)<=U(x_J); the last from binary products. This bounds the **constructed
decision objective**, not trajectory error, realized harm, coverage or physical
collision risk. Finite-precision solutions require a numerical tolerance and
checked optimality gaps. Tied solutions may select different identities.

An additive counterexample is included in tests: full geometry improves on
risk-only selection even though every d_ij is zero. A second constructed
proximity example requires a nonzero product and distinguishes the controls.
Both explain what the ablation tests; neither proves benefit on real futures.

## Numerical Contract

The versioned solver applies one positive multiplier to every objective
coefficient. It does not rescale features, redefine scores, change the primary
metric, or relax original constraints. Every arm uses that same solver version.
Binary decisions, continuous product variables, original feasibility, primal
objective and dual bound are checked. Timeout, malformed result or failed
certificate returns floor and is explicitly unmatched. Exact-count solutions
can be worse than floor: they are diagnostic, not deployment decisions.

A local SciPy1.17.1/HiGHS solve returned success with a known suboptimal witness
in a real past-input query. The completed report preserves the original result,
enumeration and controlled repair. Existing experiment-bound legacy code was
not edited. Numerical tolerance does not certify statistical or physical safety.

## Prior Art And Contribution Boundary

[JFP, Section3.3](https://arxiv.org/pdf/2212.08710) already models forecast
compatibility using unary and pairwise energies. Our binary algebra and MILP
are not novel merely because they are applied to a neural/baseline choice.
The unproven research hypothesis is whether learning baseline-relative gain and
harm, combined with composition-aware intervention and honest scene-level
calibration, provides useful predictive gains beyond matched simpler controls.

Lower proximity cost alone is insufficient. A future experiment must preserve
identical forecasts, data exposure, budgets and calibration roles, then report
accuracy and damage alongside the proxy. Support-poor or zero-count results
must remain visible. Reusing explored source sites is development evidence,
not independent confirmation. Stage5C and SMC remain disabled.
