# Post-readout query-allocation attribution

## Material Passport

This is a diagnostic registered AFTER the main allocation readout and BEFORE
the new attribution values are computed. It is not retroactive preregistration
of the primary experiment. The main method, weights, choices, cohort, risk
contract and outcome summary remain frozen. No new training or policy is made.
SOURCE fitting/resubstitution only; no independent or causal-identification claim.

## Questions and fixed contrasts

Recompute causal cost predictions for all72 original context/source/seed groups.
Verify their hashes and all four frozen action hashes against the main readout.
Read the completed private inference checkpoints; do not solve or select new actions.
The original independent arm's full legacy attribution must reproduce exactly.

For independent,uniform,joint-matched and joint-free choices:

1. Decompose observed minus predicted signed risk score into harm underprediction
   and reference overprediction, keeping all known selected rows and unknown counts.
2. Distinguish per-row observed excess from a false QUERY admission. A pooled
   controller may intentionally accept a row above2% if its complete query fits.
3. On queries whose selected rows all have observed labels, apply the same
   arithmetic decomposition to aggregated selected moments. Mixed/unknown and
   empty queries are counted separately, never imputed or used as an inference gate.
4. Delete only actual neural uses (action6 -> unchanged floor), retaining every
   other selected action. Compare the original policy with this ablation. This
   is not a reoptimized physical-only controller or a newly safe deployment.
5. Compare joint-matched versus independent neural-use effects. Where both use
   the same neural path, its contribution cancels, including on unknown rows.
   Bound the remaining unknown signed effect by the actual maximum neural-floor
   path disagreement. Do not subtract two lower bounds and call it a new bound.
6. Also compare the two neural-deleted allocations. Known utility changes split
   arithmetically into neural-use reallocation and remaining action reallocation;
   confidence-interval endpoints are not additive causal effects.

Use3,000 locality bootstrap draws after context/seed averaging and retain all
required views. Raw harm/reference sums are bookkeeping in local image units,
not cross-domain physical quantities; report within-record predicted/observed
ratios as well. Unknown outcomes remain in the full policy evaluation. Future
labels enter only this post-choice diagnostic, not inference or model selection.

## Execution and limits

Execute only after the main72-group separate inference/solver/metric replay
passes. Native arm64 CPU4/interop1/workers0,atomic group receipts,resume,heartbeat,
12h limit and10GiB disk reserve. Recompute inference and attribution, not neural
training. Keep private action caches,raw inputs and weights out of Git. Publish
only registered code,aggregate diagnostics,reports and a byte-verified metric bundle.
Any positive diagnostic is secondary fitting evidence, not a replacement for
the failed complete-policy safety gate or independent confirmation.
No metric/seconds/true3D/foundation/human-gold claim,Stage5C execution or SMC.
