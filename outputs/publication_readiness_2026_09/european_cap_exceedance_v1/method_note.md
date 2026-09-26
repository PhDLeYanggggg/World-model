# What This Probe Can Establish

## Material Passport
Analytical interpretation fixed before the new held readout. This is a
source-development diagnostic,not a replacement for the main forecasting
endpoint or the scene-joint intervention experiment.

Let h be realized nonnegative excess trajectory cost relative to the frozen
reference,and let e denote the fitting-defined easy event. The existing cost
network estimates H(x) and H_E(x),with0<=H_E(x)<=H(x)<=d(x),where d is the
causal forecast-disagreement envelope. The new supervised event is
Z=1{e*h>H(x)}. Its label is future-dependent,as supervised error labels must
be;the input x and threshold H(x) are causal predictions. At inference the
new probe receives neither e,h nor Z.

Learning P(Z=1|x) is not the same task as learning expected easy harm. In
particular,even perfect event discrimination does not identify the severity
of the omitted tail. Conversely,accurate expected harm can coexist with
unpredictable realized cap violations. The preceding fitting-only projection
used labels and was not an achievable predictor. This experiment cannot turn
that projection into a model performance upper bound without assumptions.

The four main controls answer different questions:

- Fitting-only constant prior:does a causal model improve probabilistic loss?
- Linear probe:does the small nonlinear network add predictive value?
- Causal disagreement envelope:does learned ordering beat an available geometric score?
- Frozen cost fractions:does the new label improve ranking over existing risk summaries?

All rankings use the same held event population. Nonprobabilistic geometric
scores are not passed off as calibrated probabilities. The fixed top10% is
only an ordering diagnostic,not a calibrated policy budget. Unknown future
labels remain unscored. Zero disagreement is excluded from the primary probe
because its zero-harm outcome is mechanically known.

Held binary log loss is computed on probabilities clipped to[1e-7,1-1e-7],
as fixed in the registered implementation. It is not an unbounded logit-space
loss. Brier uses the emitted probability. The recorded optimization loss uses
binary cross entropy with logits on a fixed fitting batch;these loss records
are optimization diagnostics,not independently sampled validation losses.

## Remaining Transport Confound
The fitting-row risk teacher excludes the row's locality and fits two other
localities. The outer risk teacher fits three localities and excludes the
outer locality. Both lineages are legal,but their predicted H distributions
and hence the event populations can differ. The shared three-locality easy
cut keeps event semantics fixed at the meta-fitting level,not the producer
distribution. A failed probe does not isolate feature insufficiency from
producer transport. A successful probe still needs magnitude calibration and
scene-level intervention utility before independent-data evaluation.

The objective remains stronger neural forecasting with controlled excess
loss and meaningful multi-agent joint decisions. Passing this diagnostic
alone would not meet that objective. No metric/seconds,physical-safety,
true3D,foundation,independent-confirmation or deployment claim is made.
