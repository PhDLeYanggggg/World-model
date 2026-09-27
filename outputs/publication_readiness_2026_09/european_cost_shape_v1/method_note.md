# Fixed Monotone Expected-Cost Readout

## Material Passport

Method specification and numerical checks,not a claim of held-scene benefit.
The scientific question and evaluation remain fixed by protocol.md.

## Parameterization

Let E be the causal disagreement envelope and p_A,p_E the frozen all-harm
and easy-harm predictions. Define r=p_A/E and u=p_E/p_A,using0 for zero
denominators. The parent enforces 0<=p_E<=p_A<=E. For five fixed knots let
B(r) be the linear interpolation basis. Increasing bounded ordinates theta
are represented as cumulative nonnegative increments d_0,...,d_4,with a
nonnegative slack d_5 and sum(d)=1. Prediction is cap*B(r)*theta=A*d.

For known positive-envelope fitting rows,use weights that sum to equal mass
within each fitting locality. A component minimizes

`sum_i w_i (A_i d - y_i)^2`

subject to the simplex. The constrained version also imposes

`sum_i w_i A_i d = sum_i w_i y_i`.

First cap=E and y=H_all. Then cap=the fitted all-harm prediction and y=H_easy,
using fraction u. This construction is sequential,not a joint optimum over
both components. A constant fraction is available and makes the constrained
problem feasible when the corresponding target mean is within capacity.
It does not mean every individual error can be matched or every risk ranked.

## Numerical Certificate

Each component is a small convex quadratic program. The implementation
enumerates all63 nonempty coefficient supports,solves each equality-constrained
KKT system and keeps feasible candidates,including simplex/moment vertices.
For the selected candidate d,compute

`gap = gradient(d) dot d - min_vertex gradient(d) dot vertex`.

Vertices of a simplex with one additional scalar equality have at most two
nonzero coefficients;the unconstrained simplex vertices have one. Enumeration
therefore gives a separate first-order optimality check. Scale the quadratic
by weighted squared cap to avoid units dominating the numerical tolerance.
Require gap<=1e-7 and equality/simplex feasibility,then recheck the actual
interpolated prediction's mass. This certifies the fitted finite optimization
to numerical tolerance,not the statistical quality of its predictions.

Synthetic tests compare objective values to an independent SLSQP solver for
both modes and multiple seeds,including rank-deficient and zero-score cases.
They also verify locality weighting,nested bounds,unchanged denominator
columns,unknown-label handling and absence of a target inference argument.

## Interpretation Limits

The mass constraint is empirical and training-only. It cannot imply conditional
calibration,held-locality calibration or physical safety. Frozen raw/origin-L2
controls are not exactly nested in the fixed-knot fraction parameterization.
Only the two new shape modes share the same parameterization;their easy caps
can differ because all-harm is fit first. No claim about neural representation
improvement follows from a numerical certificate or a smaller fitting loss.
