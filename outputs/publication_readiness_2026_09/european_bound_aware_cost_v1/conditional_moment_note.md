# What The New Targets Estimate

This is a direct conditional-expectation derivation,not a new proper-scoring
theorem or a verified risk guarantee. Let X be the whole available past-only
query context and fixed candidate/floor trajectories; D(X) is their maximum
path disagreement. Let nu indicate any observed requested future label. On
observed rows,B,H,R,ER,EH use the existing masked-ADE definitions.

The evaluator's contributions to its conservative completion are

    Z_B  = nu * B
    Z_H  = nu * H  + (1-nu) * D(X)
    Z_R  = nu * R
    Z_ER = nu * ER
    Z_EH = nu * EH + (1-nu) * D(X).

Thus zero R/ER on an unknown row is an absent *observed denominator contribution*,
not a claim that its true future error is zero. Direct squared cost regression
with adequate capacity and weights determined by X targets E[Z|X]. Training
only on observed rows instead fits an observation-conditioned/weighted target;
it does not automatically account for the missing-row disagreement contribution.
The old within-query observed-support normalization can additionally depend on
future label support; the new all-defined-target normalization is causal apart
from the targets. This difference is explicitly included in the matched design.

For a fixed observation mask,ADE convexity gives H_alpha<=alpha*H_1; wholly
unknown disagreement scales exactly by alpha. With ideal conditional moments,
a future-free policy satisfying alpha*m_H<=.02*m_R and
alpha*m_EH<=.02*m_ER pointwise can control corresponding ratios of expected
contributions where reference mass is positive. This is not the expectation
of a per-record ratio,nor the maximum of every observed recording's ratio.
Approximation error,selection,domain shift,finite reference support and unobserved
partial-future steps are not removed by this algebra.

The actual learned costs must therefore be assessed after their own decisions.
No population or every-record safety guarantee follows from lower fitting MSE,
the moment decomposition,head coherence or a positive observed average gain.
No current policy is promoted by this note.
