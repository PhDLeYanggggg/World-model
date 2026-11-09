# Harm conditional on nominated edits

Keep the v139 source roles, 8/12 native windows, CV fallback and risk
reference, OLS4 utility comparator, and original recording-level 2%
rule. A and all independent confirmation outcomes stay closed.

Freeze the shape response model and portfolio. In each query give
every legal row one candidate: its selected portfolio action if it
has one, otherwise its best independent action under the shape model.
Train two risk heads on this proposal distribution, balancing nominated
and alternate rows within fit sources where both are present. They
share inputs, optimizer budget, samples and pinball loss; change only
the harm quantile (0.50 vs 0.85). Targets are the nonnegative all/easy
harm masses and signed easy change. Use the same frozen gains and
reference masses for both heads. The shape forecaster can have seen
fit labels in its own training, not calibration or held outcomes.

Primary: rerank recipients at exactly the parent query counts,
without changing a row's proposed trajectory. Compare both heads
with the unchanged parent, and the tail head with the median head.
Report actual peak displacement, recipient exchanges, risks, paired
utility, and pair geometry. A free abstention ladder varies harm
weight and gain threshold on calibration; a setting must satisfy the
original risk rule in every calibration recording and positive
source-averaged lower gain against both references.

Diagnostics: best possible lower utility against each comparator
separately at the same exact query counts and fixed one-path menu,
ignoring risk; and risk prediction exceedance on the frozen parent's
nominated rows. The upper bound uses outcomes only after freezing and
is not an eligible policy or proof of joint feasibility. Estimate
uncertainty with 3,000 paired source-group resamples, averaging seeds.
