# Selected-query risk learning: method and contribution boundary

## Statistical unit

For query Q (one recording/frame), let S0(Q) be the original joint rule's causal
selected agent-action set. Unknown peers are retained. With budget rho=0.02,
the supervised signed scores are the sums of `H-rho*R` and `EH-rho*ER`, divided
by the summed causal row scale. Empty or partly unknown selected sets do not
become zero targets. Future costs enter these training labels only.

The point loss averages squared all/easy harm errors over known legal candidate
actions within each row and query. The query loss averages squared differences
of the two selected scores, using the original all-action-derived RMS. Queries
are drawn source-balanced without removing empty/unknown queries. The two terms
both have weight0.5. The later unit experiment replaces only the query-score
denominator with its actual fitting-target RMS; it is a different registered
experiment, not a post-hoc relabelling of these results.

## Why context and optimization are separate

The selected-set head uses a shared candidate encoder, mean pooling over S0 and
the focal embedding. The matched local head pairs the focal embedding with
itself. Both see selected count and fraction, start from identical weights,
receive identical training draws, and retain all forecasts and original
benefit/reference moments. Both can rescore all legal candidates, not just veto
old interventions.

Each newly fitted critic has its own independent rule and joint optimization at
that rule's count. The two full policies therefore need not switch equally
often. An additional common-minimum-count reoptimized comparison isolates a
count-matched utility contrast, not full-policy safety. The original 2% observed
recording-risk requirement is always evaluated separately.

For frozen weights, original selection is a deterministic function of the
admitted past-only inputs. The pooled context is an inductive bias; the local
features already include some neighbor/scene information. The eventual label
mask enters supervision, not pooling or inference. No missing-at-random
assumption has been established. Because references remain frozen, the signed
query target can ask harm estimates to compensate for reference bias; this
can conflict with fitting harm itself. Original-set training also does not
establish calibration after the final decoder changes that selection.

## Existing methods and what is not claimed

Permutation-invariant latent pooling is established by
[Deep Sets](https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html).
Our focal-plus-pooled encoder is a controlled use of this idea, not a new set
architecture or universality result. It adds no new causal observations.

[Decision calibration](https://proceedings.neurips.cc/paper/2021/hash/bbc92a647199b832ec90d7cf57074e9e-Abstract.html)
studies calibration for downstream decisions. The cited paper's multiclass
construction and conditions do not automatically cover our continuous costs,
coupled changing action sets or selected harm/reference ratio. We have not
implemented that algorithm or proved its guarantee for this controller. Better
MSE and predicted feasibility alone are not a substitute.

[Regression with Multi-Expert Deferral](https://proceedings.mlr.press/v235/mao24d.html)
already studies routing among pretrained regression experts and provides
surrogate consistency results. Generic forecast routing, a fallback and learned
regression costs are not novelty claims here. We do not claim superiority to
that method without a compatible matched implementation.

The useful empirical question is narrower: does a selected-query risk objective
and context repair the post-selection risk failure at retained utility? The
completed original comparison answers no in its tested form. A unit correction
can diagnose one optimization issue, but normalization alone is not a paper's
method contribution or an independent success. A credible eventual contribution
needs useful final-selection risk control, strong decision-method controls and
exposure-compliant independent evidence.

The current budget coupling does not learn dependencies between generated
future paths. [JFP](https://proceedings.mlr.press/v205/luo23a.html) studies
structured multi-agent forecasting; unary budget allocation should not be
described as equivalent dynamics. Confidence pruning also need not reduce a
selected harm/reference ratio. Existing DUT predictive exposure and reserved
DroneCrowd roles remain unchanged by this experiment.

Task: obs8/pred12 at raw stride12; detector-silver/local-image coordinates.
No new trajectory dynamics, physical safety, metric/seconds, human-gold,
true3D, foundation, deployment, Stage5C execution or SMC claim.
