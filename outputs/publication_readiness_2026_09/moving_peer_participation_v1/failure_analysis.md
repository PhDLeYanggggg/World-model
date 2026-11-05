# Factorial Peer-Context Failure Analysis

## What Was Actually Run

Six fresh neural fits complete two missing cells of a candidate-identity by
peer-participation comparison. Six previous controls are cached_verified by
checkpoint hashes and exact metric recomputation. All12cells start at the same
function within each context, have76,548parameters and1,000updates. Encoders
and bridges update. No loss, threshold, split or safety-rule change is involved.

There are62,466 own-B rows in1,087 dependent queries from eight fitting sources.
The three contexts share producer seed17. These are not independent scenes,
three independent training seeds or a new test set. All results below are
fitting diagnostics. The original24-group primary remains NA.

## Primary Result

Proposed-all versus same-index-all, with all peers retained:

| Context | MSE reduction % | Joint-cost reduction % | False-safe-cost reduction % |
|---|---:|---:|---:|
| 0 | +0.042702 | -0.043263 | +8.921983 |
| 1 | -0.279415 | -4.107580 | -19.094560 |
| 2 | -0.402056 | +0.674586 | +9.027022 |

Positive reductions are better. None meets the fixed follow-up screen: at
least1% lower joint cost, nonworse MSE and nonworse false-safe cost. Context2
reduces joint cost slightly but misses the effect requirement and worsens MSE.
Context0 reduces harmful admission but slightly increases missed beneficial
admission cost. Context1 worsens all three. No extra nested-C/H readout ran.

## Which Hypothesis Survived?

Adding fallback peers under proposed-candidate context changes joint cost by
-0.971716%,+2.680854%,+1.226506% across contexts0/1/2. Thus omitted background
can explain part of the previous degradation, but it is not a sufficient repair.
Retaining peers is not uniformly beneficial under the same-index representation
either: joint-cost reductions are-0.993545%,-8.203737%,+0.308604%.

With participation fixed to active peers, changing candidate identity worsens
joint cost in all three contexts:-0.064891%,-15.751419%,-0.248445%. Context1
also illustrates why selecting the lowest total fitting cost alone is unsafe:
same-active lowers total cost versus same-all but increases false-safe cost.
No favorable secondary cell is substituted for the registered primary pair.

The result does **not** establish that same-index context is semantically
correct for a final policy, that proposals are useless, or that context can
never help. It rejects a specific mean-pooled bridge repair under this budget.

## Failure Slices

Context1's primary joint-cost increment is0.000111849044 in normalized fitting
units. Source065 contributes0.000108856591, or97.3246% of the net increment.
Its conditional cost rises0.001149357to0.001940163. The previous proposed-active
value was0.002198945: the repair helps that failure but does not remove it.
These weighted source slices are posthoc diagnosis, not a source exclusion rule.

Context1's single/empty-proposal conditional cost changes0.001003808to0.001691165;
mixed-proposal cost changes0.003652115to0.003729787. Empty queries have zero
active objective weight and never become a safety certificate. Both query
categories still worsen against the registered control.

Context2's total improves in all three fitting sources, but the mixed-query
improvement coexists with worse single/empty-query cost. Uniform improvement
over all sources, query types and risk channels has not been demonstrated.

## Is The New Context Actually Different?

On each parent's causal proposed rows, before the new bridge is trained:

| Context | Proposed-all vs same-all mean cosine | Relative RMS difference |
|---|---:|---:|
| 0 | 0.998376 | 0.057468 |
| 1 | 0.990964 | 0.131042 |
| 2 | 0.994923 | 0.108309 |

The representations are highly aligned but not identical. The measurement is
unweighted descriptive geometry on the shared parent risk encoder; zero-norm
cosines are NA, not zero. It cannot prove information sufficiency or identify
mean pooling as the causal reason for failure. Learned bridge norms are nonzero;
the result is not an accidentally frozen encoder or inactive branch.

## Execution And Limits

- Pilot:500actualupdates, both new-arm resumes exact, legacy100-update engine exact.
- Primary:6,000newupdates plus1,000exact full-refit updates; PID36291 exits0.
- Training778.812s, peak RSS7,301,939,200bytes; no resource blocker or downgrade.
- All12initial/final metrics and attributions replay exactly;44focused tests pass.
- Diagnostic PID37605 exits0 in38.610s; prior public files remain unchanged.
- Loss/cost figure visually inspected. The historical full suite was not rerun.
- Exact replay verifies execution, not independent scientific confirmation.
- No new held ADE, calibrated risk pass, deployment, JEPA or dynamics claim.
- Original2% all/easy harm, easy-net and support rules remain. No identity exemption.
- Original24-group primary NA; Moving20, DUT, DroneCrowd and H remain closed.
- Native obs8/pred12 silver image coordinates; no metric/seconds/true3D/foundation.
- Stage5C/SMC remain disabled; not yet submission-ready.

## Next Decisive Question

Stop treating inclusion of fallback peers as a sufficient fix. The next model
question is whether ego-candidate-relative peer information has source-held-out
predictive value beyond the existing32relational features and query mean. First
check the earlier negative16-probe neighbor-feature study and query-context
studies; do not rebrand them as a new experiment. A bounded own-B, source-held-out
information test should compare the same supervision and causal candidate bank,
including source065, before investing in another larger pooling architecture.

If that information test fails, inspect label completeness and conditional
predictability in the permitted training sources rather than repeatedly changing
context masks. Missing reference support cannot be repaired by a network or by
renaming an empty estimate. None of these next steps opens reserved evaluation
or alters the user's original rules.
