# Paired Decision Learning Under A Fixed Risk Contract

## Question And Controlled Method

We test whether within-query discrimination between harmful and beneficial
forecast interventions improves a fixed acceptance policy. Pairwise logistic
ranking is established; this experiment is not a new ranking-loss claim.
Three arms use the same 64,260-parameter neural risk/decision model, 492 causal
inputs, warm initialization, query sequence, preprocessing and 2000-update
budget: the preceding joint objective, a matched paired-point objective and
a paired-rank objective. Pairs contain observed supported nonzero targets
from the same causal query. Unknown-label supervision stays unchanged.

The point and rank terms share value and score gradient at zero, but do not
have identical realized gradients throughout training. Rank is invariant to
a shared score shift; the existing supervised terms anchor absolute scores.
All nine fits are frozen before readout. The primary contrast is rank minus point;
there is no posthoc choice of arm, cutoff, support rule or risk tolerance.

## Results

Ranking improves fitting AUROC over joint in all three contexts but has higher
total decision cost than point by 46.55%, 19.67%, 1.73%. Point admits more useful
predictions and more harmful predictions. On seven exposed development sources,
known partial-outcome ADE gains are 0.8948%, 1.4674%, 0.8363% for joint, point, rank.
The primary rank-minus-point effect is -0.6311 pp, with missing-label completion
bounds [-0.8664, -0.3957] pp. These are not sampling confidence intervals.
Six of seven source contrasts are negative. Rank's worst defined easy-net
degradation is 8.3373%; all three policies fail the complete original risk/support
requirement, with 0/17 recording views satisfying it. No model is promoted.

## Interpretation And Limitations

The experiment rejects this specific ranking repair. It does not show that
all pairwise objectives fail or that relative-risk learning is impossible.
Improved ordering is insufficient to identify useful interventions at the
fixed acceptance boundary. Greater average utility alone is also insufficient
under the original safety contract. Both actual risk violations and missing
support are reported, without an identity exemption.

This is repeated development with one forecast-producer seed and partial
outcomes. Three fitting contexts are not three independent replications.
Exact recovery and replay establish engineering reproducibility, not sampling
confidence or independent confirmation. The original 24-group primary remains
NA; reserved confirmation data stay closed. No new trajectory predictor,
JEPA contribution, physical-safety theorem or deployment success is claimed.

## Reproducibility And Scope

The record includes nine full fits, three exact predecessor controls, one exact
full refit, three interrupted-versus-continuous pilots, 66 focused regression cases and full
prediction/metric replay. Code, configuration, registration, summary metrics
and scientific SVG plots are public; private data and weights are excluded.
See [validation](validation_receipt.md), [failure analysis](failure_analysis.md)
and [method positioning](related_work_position.md).

Native obs8/pred12 image-coordinate results are not metric, elapsed seconds,
true3D, foundation or human gold. Stage5C/SMC remain disabled. This addendum
is separate from the compiled main manuscript; linking it does not update
that PDF or establish submission readiness.
