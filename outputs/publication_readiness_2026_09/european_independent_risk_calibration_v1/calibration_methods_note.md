# Empirical Calibration And Method Positioning

Working methods text, before calibration outcomes. Results are not inserted
until the fixed-grid run and original risk checks complete.

## Frozen Multi-Action Decision

For agent i and available action k, the SOURCE-trained critic predicts native
benefit B, positive harm H, reference error R, easy-reference ER and easy harm
EH. Forecasts, normalization and SOURCE easy/hard cuts stay fixed. Native
utility is G=B-H. No target or eventual label mask enters this decision.

At a contemporaneous query, joint allocation maximizes the sum of G over
binary available-action choices, with at most one intervention per target.
For risk budget rho=.02 and a common multiplier lambda, the two constraints
are sum(lambda H-rho R)<=0 and sum(lambda EH-rho ER)<=0. Matched allocation
also fixes the intervention count to the independent policy's count at the
same query and lambda. This is pooling predicted error budgets, not optimizing
physical pairwise collision constraints or claiming all-visible-agent control.

Implementation divides only the R and ER optimization coefficients by lambda.
G and the forecast bank remain exactly unchanged. The transformed coefficient
array is not a coherent new conditional cost distribution or calibrated
posterior. It is an equivalent way to tighten the two optimization constraints.

Freeze all four policies and all 13 binary multipliers for every pipeline and
recording before constructing target labels. The common calibration parameter
is the least registered lambda satisfying every original recording view for
both required policies. Nonestimable support does not pass. No calibration
utility criterion selects lambda, model, seed or SOURCE fold.

## Existing Risk-Control Work

[Regression with Multi-Expert Deferral](https://proceedings.mlr.press/v235/mao24d.html)
already studies continuous regression-cost routing, including a two-stage
setting with pretrained experts. Our earlier matched published-loss control
adapts its logistic surrogate; it is not a new deferral loss or a reproduction
of the authors' original experiments. [Optimized Deferral for Imbalanced
Settings](https://proceedings.mlr.press/v306/cortes26b.html) studies cost-sensitive
expert imbalance. MILD has not been reproduced here, so imbalance alone cannot
be presented as a demonstrated methodological contribution.

[Learn then Test](https://arxiv.org/html/2110.01052v5) already provides a
hypothesis-testing framework for risk calibration, including selective
regression. Valid risk tests and multiplicity control matter; inspecting a
finite grid alone does not implement its guarantee. This study does not claim
that risk-aware selective regression is new.

[Conformal Risk Control](https://arxiv.org/html/2208.02814v4) treats exchangeable
bounded loss curves, and also discusses monotone envelopes for nonmonotone
losses. Nonmonotonicity alone is not a claim that conformal methods are
impossible. Our empirical recording-ratio selector has not implemented that
envelope, bounded-loss construction or the required inferential procedure.
We therefore do not call this operator conformal or claim its formal guarantee.

## What The Experiment Can Establish

The experiment asks whether a single risk-only adjustment repairs the frozen
cost family on a separately reserved role, without changing utility ranking,
action bank or the original support conditions. Its limited scientific value
is distinguishing calibration error from ranking/representation or label-
support failure. Passing empirical calibration would still require independent
confirmation before a main efficacy claim; failing it motivates a specific
SOURCE-side repair rather than another exposed-scene threshold search.

Inference/statistical units remain separate: 102 recordings are grouped into
12 localities, and nine pipelines create repeated views. Utility summaries
pool native reference mass within each locality and weight localities equally;
conditional locality bootstrap intervals do not undo calibration selection.
Overlapping windows are not independent trials. Geographic grouping does not
by itself prove statistical independence or online sensor causality.

Obs8/pred12/raw stride12, local image pixels and released detector/tracker
silver remain the protocol. Available-label ADE and wholly unknown-row bounds
do not recover missing steps in partial trajectories. No metric, seconds,
true-3D, foundation, physical-safety, Stage5C or SMC claim follows from this work.
