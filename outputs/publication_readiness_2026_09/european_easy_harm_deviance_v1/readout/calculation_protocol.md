# Fixed Development Calculation Contract

Result status: **not_run**. This document freezes the evaluation calculations
while the full TRAIN experiment is incomplete. No new development prediction
has been scored. The parent training protocol and its scientific targets stay
unchanged; this is not an execution waiver for the failed control.

## Arms And Decision Rule

Keep all six arms: original forest, additive quality control, positive-harm
Poisson-style control, cost control, matched quadratic neural head and
easy-harm deviance neural head. Frozen trajectory predictions, causal features,
support masks and the2% risk budget are unchanged. Actions use predicted costs,
moving/support flags and past-only context, never evaluation-label availability.

Report full decisions and pairwise same-count decisions for deviance versus
each of the five comparators. Match within each recording/frame, using predicted
gain and stable row IDs to break ties. Do not match only a global action count.

## Costs And Missing Outcomes

Retain known partial-label costs as observed. Wholly unknown outcomes remain in
the inference population. For paired utility, shared unknown actions cancel;
only exchanged unknown actions contribute their causal disagreement envelopes.
The difference of two separate lower bounds is a secondary proxy, not the paired
lower utility. Undefined denominators or empty cohorts remain undefined.

Report signed gain/all-risk/easy-risk error on all known rows, the original
policy's selected cohort and each arm's selected cohort. Include harm/reference
mass, predicted/observed selected easy-harm ratio, recording-level completion
bounds and normalized selected-harm tail summaries. None is relabeled as FDE
improvement or a physical-safety guarantee.

## Prespecified Decision

The four primary contrasts are deviance minus quadratic and original forest,
each for full and same-count paired lower utility. Average seeds and views
inside the12 source localities, then use3,000 paired locality bootstrap draws
with registered seed20261006. These are nominal exposed-development intervals.
Do not count72 source/head views or overlapping windows as independent scenes.

Advancement requires all four lower interval limits to be positive. The full
deviance policy and both primary same-count policies must each retain finite,
nonempty completion support in every view and have no known or completion-upper
easy-risk violation. In particular, empty matched support does not earn a pass.
Secondary MSE is reported but does not replace the registered utility/safety
screen. The remaining strong controls and negative comparisons stay visible.

Even a pass only permits later transfer-design work. It does not authorize
independent calibration/test access, deployment, metric/seconds claims, Stage5C
or SMC. Historical test-tuned results remain exploratory.

## Verification Scope

The implementation has separate scalar recomputation of decisions, query
matching, completion costs, error weighting, tail/mass summaries and locality
bootstrap. Mutation tests check that corrupted risk, actions, paired utility or
harm summaries fail. This synthetic verification is not a real-data readout.

Before real inference, all144 final checkpoint receipts must be verified and
frozen, all72 original quadratic controls must pass, and the scheduler join
must complete successfully. The current partial36-fit record is insufficient.
