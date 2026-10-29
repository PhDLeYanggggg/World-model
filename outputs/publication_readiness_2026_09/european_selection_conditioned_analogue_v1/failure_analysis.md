# Failure Taxonomy: Population Matching Is a Partial Repair

## Supported Findings

| Mechanism | Observed evidence | Interpretation |
|---|---|---|
| Reference population mismatch | Changing label donors only reduces easy/all risk MSE50.10%/65.79%; normalization and old controls exact | The bank construction matters. It is not the sole cause, and fewer donors are not automatically more representative. |
| Reference scarcity | Minimum393 selected donors, all62,902 queries supported | Insufficient k32 support is not the failure in this run. |
| Aggregate event frequency | All event probability19.7416% versus prevalence19.8027%; easy6.2746% versus6.1335% | Aggregate event frequency can be close while conditional risk remains poor. |
| Actual-event harm magnitude | Easy event-positive SSE39.001010 ->40.662720; predicted positive-event mass7.230221 versus observed95.560642 | Lower overall error does not show improved estimation of actual harmful cases. |
| Non-event excess prediction | Easy non-event SSE82.241591 ->36.437111 | This is where much of the harm-SSE reduction occurs. It is not proof that all non-event probability should be zero. |
| Incremental matching information | Selected nearest easy MSE0.002365719 versus mean0.001709022 and shuffled0.002384333 | The tested Euclidean neighborhood adds no favorable primary MSE result beyond the selected mean; it only slightly beats shuffled. |
| Original selected-cohort comparator | Nearest easy MSE7.65x original;0/18 pair wins against both required comparators | Registered magnitude gate fails. Original's low MSE is also not a safety guarantee; its aggregate harm remains severely underestimated. |
| Source heterogeneity | Locality020 dominates60.67% of equal-view easy MSE; locality048 and one context worsen | No source deletion. Coordinate/rollout-envelope transfer and local conditional labels remain possible contributors, not proven causes. |
| Unknown and rare events |1,567 unknown queries retained;346 positive easy events without positive nearest donor | Cannot infer safe risk from missing outcomes or zero positive neighbors. |
| Joint output compatibility | Selected mean/shuffle each2 B+H envelope violations, nearest0 | Separate moment replacement is not automatically a coherent deployed controller. No projection or new policy was used. |

## Rejected Shortcuts

- A single global shrink factor would lower already underestimated harm on
  observed positive events; no fitted global rescale is authorized by this result.
- A better nearest-versus-shuffle comparison alone cannot replace the original
  neural comparator or the approved positive-harm risk criterion.
- Locality020 is not removed; a clean-looking average after removal is not the
  primary result. Its limited query support is an explicit source limitation.
- Mean easy-event probability is not calibration conditional on selection,
  risk magnitude, source or future intervention. Easy event here means easy-H>0.
- The Brier and harm-SSE tables use different targets and aggregation weights
  from equal-view signed-risk MSE; their percentage changes are not interchangeable.

## Next Test and Its Limit

The current estimator averages H/envelope uniformly, then multiplies by the
query envelope. The evaluated quantity is instead source-normalized signed
risk with a copied reference moment. These objectives need not have the same
local optimum. Before a new run, reconcile existing cost-aligned Newton results,
derive the exact local quadratic cost objective and register a fixed-neighbor
selected-bank comparison. This would test loss alignment without changing
queries, features, thresholds or the2% standard, rather than repeating global
hurdle training or another population-calibration sweep. It has not run.

Even a fitting improvement would still require genuinely independent risk and
policy evaluation, useful matched intervention comparisons and new forecaster
replicates before a paper claim. Upstream exposure is not erased by reference
exclusion. No neural training, deployment, Stage5C or SMC this round. Pixel/local
rawstride12 obs8/pred12 detector-silver only, not metric/seconds or human gold.
