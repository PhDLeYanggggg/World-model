# Concentrated Harm Is Not Reliably Identifiable Across Sources

## Completed Comparison

Eight positive-target regressors are fresh_run; eight signed-risk models are
cached_verified controls. Both sets reproduce by a full refit. The trajectory
producer, frozen proposals and prior neural heads are not retrained here.
Same120causal inputs,64trees/depth8/minleaf32 and source/query weights are used.
Positive targets also induce their own fitting-only RMS scaling under the
inherited learner; this is not a pure cancellation-only or fixed-loss-scale
ablation. No probability calibration or upper-risk guarantee is supplied.

At the fixed10% ranking weight, source-macro easy positive-excess capture is:

| Context | Signed-risk control | Positive-tail target | Envelope | Label oracle |
|---|---:|---:|---:|---:|
| 0 | 19.196% | 20.675% | 12.214% | 99.880% |
| 1 | 3.419% | 4.078% | 20.427% | 99.990% |
| 2 | 38.228% | 39.099% | 32.400% | 98.798% |

The changes over the learned control are only+1.479,+0.659,+0.871percentage
points. The fixed information screen fails0/3. This does not support expanding
the positive-target branch into a larger neural model, nor changing its
threshold to manufacture a successful selection result.

## What The Tail Audit Establishes

1. Easy positive-excess mass is highly concentrated. An outcome-sorted
   diagnostic ranking captures98.8-100% in10% of query-balanced weight. This
   is not a realizable oracle selector, independent evidence or model accuracy.
2. Most mass is associated with full-amplitude neural proposals:93.036%,
   53.442%,84.346% across contexts. Their weight shares are88.382%,22.167%,
   43.852%. The latter two contexts have disproportionate mass at full amplitude.
   This is confounded by which cases the frozen proposer chooses. It does not
   establish that uniform shrinkage fixes safety; prior amplitude experiments
   already failed that stronger claim.
3. Source transfer is weak. The positive-target model captures60.2-86.9% on
   its fitting sources, but0-49.7% on excluded sources. In sources053 and065
   it captures0% at the fixed budget. Source065 has only44known proposals and
   10easy-reference rows, not a large independent population. No source is
   removed to improve the macro result.
4. Source motion ranges differ markedly. Using other-source causal quartile
   cuts,93.3% of source048's weight is in the lowest speed bin, whereas92.3%
   of source067 is in the highest. Source059 has88.8% in the highest
   acceleration bin, while source073 has83.5% in the lowest. These are
   causal-normalized image-coordinate quantities, not verified speeds or
   physical acceleration. Source-specific causal cuts and all bins are retained.
5. Plain neighbor count is nearly saturated: seven of eight source folds
   place all rows in the same tied upper bin. That does not show interactions
   are irrelevant; it shows why this coarse count is a weak discriminating
   variable for this particular admitted population.

## A Concrete Representation Suspect, Not A Proven Cause

The existing `causal_cost_features` divides ego motion and candidate paths by
the maximum of1pixel, ego-history radius and all valid neighbor-history radii.
The denominator therefore changes with the outer neighbor geometry even when
ego motion is unchanged. This is causal, not leakage, but can compress the
ego-motion representation differently across sources. The observed range
shift motivates a matched ego-history-frame representation control. It does
not yet prove that this normalizer causes the held-source failure, or that a
new normalization will help. Camera motion, tracking/label quality, producer
error, small source support and conditional-outcome differences remain plausible.

The next experiment should preserve the original signed targets, output units,
risk limits and support requirements while changing the causal motion frame.
Retain the old-to-new scale ratio as an explicit causal feature to avoid
discarding context or silently changing the estimand. Compare against the
existing same-capacity control before adding neural capacity or new modalities.
Do not turn these exposed source diagnostics into another held-test search.

## Boundaries

All48,061known proposals and1,274unknown proposals are accounted for. Eight
source folds are not eight independent producer seeds. Overlapping windows
are not independent samples. Attribution and ranking use known available
labels; missing outcomes are not recovered. The original24-group primary
remains NA, not passed. No independent C/H or reserved confirmation is opened.

No new policy was evaluated: ranking at10% with fractional tie credit is not
a deployed intervention policy. No original safety gate passes by this result.
The old2% all/easy harm, easy-net and support requirements stand; empty controls
remain NA; no identity certification. Old negative studies remain negative.
No deployment upgrade, Stage5C, SMC, true3D, foundation, metric/seconds claim
or submission-ready claim. Current task is native obs8/pred12 image coordinates.
