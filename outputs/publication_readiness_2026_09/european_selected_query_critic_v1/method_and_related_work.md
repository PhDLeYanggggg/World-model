# Selected-set risk learning: question, assumptions and prior work

This is a method-design component, not a successful result or independent test.
The registered protocol fixes the paired experiment; this note does not change
it. Main task: obs8/pred12 at raw stride12 in local image coordinates.

## What is learned

The original action bank and benefit/reference/easy-reference estimates are
fixed. Two equal-size neural heads adjust positive harm and positive easy harm
through the same bounded anchor-exact link. Both train on all legal candidate
cost labels plus the signed score of the original joint selected set. The
only paired architecture difference is self versus mean selected-action hidden
context. Both see the same initial count metadata.

The resulting system is a risk-learning controller around fixed forecasts. It
is not newly trained trajectory dynamics, a generative JEPA model, a world-state
rollout or a foundation model. A favorable new neural risk head cannot substitute
for evidence that the underlying world representation/dynamics contributes.

## Conditioning is an inductive bias, not new information or a theorem

For frozen weights, the original selected set is a deterministic function of
the admitted past-only query inputs. Its pooled representation does not add
an independent observation or reveal a future label. It is a structured way
to expose other selected action histories and costs to a row-level risk head.
The row-local comparison is needed because the existing causal fields already
contain local neighbor information and some scene features.

The pool includes selected rows with unknown eventual labels. That unknown
mask enters only supervised loss/evaluation, never pooling or admission. Missing
selected labels prevent a complete-query target, not causal inference. Training
therefore has a missing-support limitation; no missing-at-random assumption
has been established.

The signed-score target contains true reference costs, although those reference
predictions remain frozen. This can create tension between matching actual
harm and compensating for reference bias. Integrated component-gradient traces
are intended to expose such conflicts rather than infer them from one endpoint.

There is another deliberate limit: the query term supervises the original
selected set, but the new decoder may reassign actions. All legal actions have
pointwise supervision, yet this does not establish calibration under the final
selection. The paired study tests a first selected-context repair; it does not
claim to solve policy-induced conditional calibration in general.

## Relation to primary literature

[Deep Sets, NIPS2017](https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)
develops permutation-invariant set models. Mean pooling a selected set is an
ordinary set-model operation, not an architectural novelty claim here. Our
controlled question is whether this particular selected-action context improves
cost decisions relative to an equal-size local head.

[Calibrating Predictions to Decisions, NeurIPS2021](https://proceedings.neurips.cc/paper/2021/hash/bbc92a647199b832ec90d7cf57074e9e-Abstract.html)
introduces decision calibration for downstream decision-makers in multiclass
prediction. It is relevant to the distinction between marginal forecast fit
and useful decisions. We do not implement that paper's recalibration algorithm
or inherit its guarantee for our continuous, selected-reference ratio.

[Regression with Multi-Expert Deferral, ICML2024](https://proceedings.mlr.press/v235/mao24d.html)
analyzes choosing between regression experts, including a pretrained-predictor
two-stage setting. Our fixed-bank controller lies within that broader routing
motivation. Calling a head cost-aware or using multiple experts is not new by
itself. Distinctive evidence would have to concern actual gain/harm behavior,
query allocation, support and source-excluded/independent risk assessment.

The current query-budget decoder does not model interactions between generated
future paths. [JFP, CoRL proceedings2023](https://proceedings.mlr.press/v205/luo23a.html)
already studies structured dependencies among multi-agent forecasts. Unary cost
budget coupling should not be described as equivalent joint dynamics learning.

## What would count as evidence

Fresh real training, a lower loss and exact replay establish implementation
and optimization evidence. Selected-set versus row-local full and common-count
comparisons address the pooling hypothesis. The original complete-recording
all/easy2% criterion still governs safety; confidence pruning alone is not
monotone in selected harm/reference risk, as the prior tested counterexample
shows. Empty selected support is undefined, not a safe zero.

All current source material is exposed development/fitting. Neither the three
critic seeds nor the locality bootstrap converts it into independent scientific
confirmation. A model must first justify its risk behavior, then undergo a
separate exposure-aware independent protocol. Historical DUT predictive exposure
and reserved DroneCrowd roles remain unchanged.

No metric/seconds, human-gold, true3D, foundation, deployment or submission-ready
claim is made. Stage5C execution and SMC remain disabled.
