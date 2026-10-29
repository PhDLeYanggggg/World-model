# What Convex Amplitude Transport Does And Does Not Guarantee

For a fixed query, let B be the causal floor path, P an endpoint candidate,
Y a future target and M a nonempty fixed evaluation mask. Define

```text
L_M(P) = mean_{t in M} ||P_t - Y_t||_2
P_alpha = (1-alpha) B + alpha P,     0 <= alpha <= 1
g = L_M(B) - L_M(P)
Bcost = max(g, 0),                 Hcost = max(-g, 0).
```

Convexity gives L_M(P_alpha) <= (1-alpha)L_M(B) + alpha L_M(P),
so g_alpha >= alpha*g. Hence Bcost_alpha >= alpha*Bcost and
Hcost_alpha <= alpha*Hcost. Multiplying by the same fixed easy-event
indicator preserves the harm inequality. This is elementary convexity,
not a new learning theorem or a bound on the benefit of a learned policy.

If endpoint estimates were the correct conditional moments under exactly
the same input information, their scaled benefit/harm would provide useful
one-sided conditional surrogates. The experiment does not assume that premise
has been established. Current learned heads are demonstrably miscalibrated
in their selected regions. Multiplying a wrong estimate does not make it a
risk certificate. The reference and easy-reference moments are unchanged;
zero or unidentified denominators remain zero or unidentified.

Actual fractional benefits can be strictly greater than scaled endpoint
benefits. A floor and endpoint equally far from the target can have a perfect
midpoint. Therefore observed endpoint errors cannot simply be scaled to
report fractional performance. The scoring code recomputes native errors
using the original target and exact mask, retaining wholly unknown rows.

For partial or wholly unknown labels, the maximum per-step disagreement
between the actual two paths bounds any masked-mean loss difference. The
original canonical risk envelope also includes disagreement to CV, and is
recomputed after interpolation; it need not equal alpha times the old
CV/floor envelope. The separate paired-utility bound uses the actual paths
being compared. Neither operation reveals future support to the decision.

Alpha is a fixed dimensionless output coefficient, not a time unit, physical
validity certificate or learned dynamics variable. The same-set control
changes outputs without allowing a new learned selection region. The
reselection control additionally admits actions according to transported
cost surrogates. Their difference must not be attributed solely to better
neural dynamics. Both retain all recordings and the unchanged 2% contract.
