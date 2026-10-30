# Reoptimized-bank control: definitions and interpretation

This is a fixed-predictor experiment, not a new dynamics model. Its purpose is
to remove a weak physical-only comparator from the attribution argument.
Deleting a chosen neural trajectory to the fallback is a useful forecast-use
ablation, but does not allow the physical controller to reconsider its actions.
The present control does allow that reconsideration.

## Common feasible cardinality

For each current recording/frame query Q, compute the frozen pointwise rule
using only the six physical candidates. Let its selected count be k_Q. Keep
the original causal eligibility and all/easy predicted risk budgets rho=0.02.
The physical assignment is feasible in both the physical and expanded banks.
No future label determines k_Q. Queries with k_Q=0 remain in every score.

Both joint problems maximize predicted net gain under at most one action per
target, exactly k_Q interventions and the two signed budget constraints:

```text
sum selected predicted_harm - rho * selected predicted_reference <= 0
sum selected predicted_easy_harm - rho * selected predicted_easy_reference <= 0.
```

The six-action feasible set is contained in the seven-action feasible set.
At exact optimum the expanded bank cannot have smaller predicted utility.
The implementation also preserves this nesting if numerical postchecks fail:
its neural warm fallback is the better predicted-gain feasible assignment from
physical joint and trimmed neural independent. Nonoptimality remains reported.
This nesting is about the fitted objective, not observed accuracy or safety.

Equal intervention counts do not enforce equal selected reference mass. Both
arms obey the same ratio criterion, not an identical absolute harm allowance.
No label-aware best-baseline choice is an inference input.

## Four controls, not a claim of causal factorial identification

Write the four complete-policy observed error masses as E_PI, E_NI, E_PJ,
E_NJ, for physical independent, trimmed neural independent, physical joint,
and neural joint. The primary known contrast is E_PJ-E_NJ, divided by the
unchanged known full-floor error mass R. The descriptive interaction contrast
is `(E_PJ-E_NJ)-(E_PI-E_NI)`, using that same R exactly once.

For a wholly unknown row, let d(a,b) be the maximum requested-step Euclidean
distance between the two chosen forecast paths. The triangle inequality gives
`abs(e_a-e_b) <= d(a,b)` for any common ground-truth path under ADE. Therefore
the four-policy interaction is bounded in absolute value by either pairing:

```text
d(PJ,NJ) + d(PI,NI)
d(PJ,PI) + d(NJ,NI).
```

Use their rowwise minimum before summing. It is valid because each is a bound
on the same row contrast. Identical shared forecast terms can cancel. The
completion-lower contrast is the known signed mass minus this unknown bound,
then normalized by R. It is not subtraction of two independently computed
lower bounds. The main paired bank contrast has its own two-policy bound.

This is forecast-error arithmetic, not an intervention on real pedestrian
behavior. It does not establish learned interaction dynamics or causal effects.
Known costs on partially observed ADE rows remain held fixed; the bound does
not certify missing interior positions in those partial labels.

## Why this cannot certify deployment

The [preceding method note](../european_frame_action_allocation_v1/method_note.md)
defines the original complete-recording known and unknown-outcome risk checks.
They remain unchanged. Predicted budgets compose algebraically across disjoint
queries; predicted costs need not be calibrated after joint selection. Even
exact conditional moments would not alone imply compliance in every finite
recording. Undefined selected easy reference is not a zero-risk certificate.

The source queries, teachers, cost heads and development history are exposed.
Locality bootstrap and three head seeds quantify conditional development
variation, not independent efficacy. Future calibration/confirmation require
their own exposure-aware admission and a frozen method. Existing DUT predictive
exposure is recorded; no DUT or DroneCrowd role is opened by this comparison.

Joint forecasting and selective routing are established topics, as illustrated
by [JFP](https://proceedings.mlr.press/v205/luo23a.html) and
[multi-expert regression deferral](https://proceedings.mlr.press/v235/mao24d.html).
Neither a seven-action bank nor a mixed-integer solver is itself a novelty claim.
An eventual method contribution needs a demonstrably useful risk mechanism and
independent evidence, not just a better fitting point estimate.

Task: obs8/pred12, raw stride12, image-local detector-silver trajectories.
No metric/seconds-level, human-gold, true3D or foundation claim. No Stage5C
execution, SMC or deployment change.
