# Experimental Addendum: Matched Contemporaneous Context

We tested whether query-wide causal information improves conditional harm
learning beyond an agent-local cost head. Keeping upstream forecasts, benefit
and reference moments fixed, we compared two equal-parameter risk branches:
one encoded the focal descriptor twice, while the other combined it with the
leave-self-out mean descriptor of agents available at the same recording/frame.
The mean was computed before encoding and did not use outcome availability.
Both branches already had access to the existing local-neighbor features.

We trained36 heads across six producer/controller contexts and three head seeds,
using identical initializations, queries and 2,000 updates per fit. A copied-
encoder trainability defect was detected and repaired before any held outcome
readout; the defective runs were retained separately and excluded from this
comparison. All corrected encoders changed while protected outputs remained
exact. The complete family was frozen before evaluation.

On72 original-TRAIN internal held-recording views, query-wide mean context
reduced matched-intervention paired completion-lower utility by0.239pp relative
to the self control (nominal95% locality-bootstrap CI[-0.353,-0.144]). Complete
risk support remained0/72. At equal counts, lower known benefit and higher
known harm explained-0.079pp; uncertainty about exchanged unknown outcomes
accounted for an additional-0.161pp. Thus uncertainty was not the sole reason
for the negative result. No favorable seed or source was selected afterward.

This is a negative test of a particular mean-pooling construction, not evidence
against all relational modeling. The intervals use twelve repeatedly exposed
TRAIN localities and are not independent confirmation or search-adjusted
inference. Head seeds are not forecaster seeds. No new ADE/FDE, physical safety,
metric geometry or world-dynamics contribution is claimed. Independent roles,
deployment, Stage5C and SMC remain unchanged. The full contrast table, numerical
provenance and limitations are in conclusions.md and failure_analysis.md.
