# Draft Diagnostic Paragraph, Not a Main Result Claim

We examined whether frequent gradient clipping explained the failure of
query-context risk estimation. Using36 frozen heads and72 original-TRAIN
fitting views, we evaluated both architectures on identical causal cohorts.
On the frozen self-selection cohort, easy-risk signed residual MSE increased
from0.000684616 to0.000728945 with peer context despite improved aggregate
harm mass. Near-zero easy output/logit derivatives persisted on75.10% and
67.95% of repeated occurrences. Across288 fixed next batches, removing clipping
from one disposable update more often reduced the scalar training objective
(249/288), but improved fixed-cohort easy-risk MSE in only82/288, worsened it
in74 and was tied in132. This motivates separating optimization of the policy
surrogate from conditional cost estimation, without establishing that a new
optimizer or architecture would resolve the problem.

These are dependent fitting diagnostics, not independent samples, confirmatory
confidence intervals, forecasting gains or safety guarantees. Unknown outcomes
are retained in inference/derivative accounting and excluded only from supervised
residual calculations. No persistent model or deployment change was made.

## Methodological References

Norm clipping rescales a gradient above its threshold; its activation frequency
alone does not specify the downstream parameter update. Our counterfactual
retains the adaptive optimizer's saved moment state, rather than treating the
clipping factor as a learning-rate ratio. This is a diagnostic design choice,
not a novel optimization method. See the original treatments of
[gradient norm clipping](https://proceedings.mlr.press/v28/pascanu13.html) and
[Adam's adaptive moments](https://arxiv.org/abs/1412.6980).

The RNN setting in the clipping paper is not empirical evidence for M3W, and
Adam is cited for moment-based updates, not as a claim that it removes all
effects of clipping. The actual experiment uses AdamW with saved weight decay.
