# Method Positioning

This is a controlled neural-risk diagnostic within M3W, not a claim that a
multi-head architecture is new. The prospective protocol is unchanged; this
note adds a closer reading of published formulations after registration.

SelectiveNet jointly trains prediction, selection and an auxiliary prediction
head on a shared representation. Its objective minimizes selective risk with
a coverage penalty; the auxiliary task exposes the representation to the
whole fitting population. Its coverage calibration must not be interpreted as
M3W's relative-baseline harm guarantee. See Sections2,4,5 of
[Geifman and El-Yaniv, ICML2019](https://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf).

Learning to defer treats the expert as an additional action and gives a
cost-sensitive surrogate and consistency analysis. Joint learning and
cost-aware rejection are therefore established ideas. The population
consistency statement does not certify a finite, source-shifted M3W neural
head with missing outcomes. See Sections3,4 of
[Mozannar and Sontag, ICML2020](https://proceedings.mlr.press/v119/mozannar20b/mozannar20b.pdf).

## What This Experiment Actually Tests

The trajectory producer and its seven-action proposal mechanism remain frozen.
For each legal proposal, the model predicts two signed risk channels, all and
easy. A second head learns cost-weighted signs for those same channels. The
fitting labels remain the original relative-baseline budget-excess targets,
including conservative bounds where future outcomes are unavailable.

The narrow and wide controls share their causal-query representation between
the two tasks; the treatment has separate representations. All use the same
source-separated fitting data, preprocessing, query draws and fixed updates.
Complete current-query other-agent pooling is retained, not replaced by new
attention or a newly trained interaction model.

At inference, both learned signs must accept the proposal, followed by the
original support requirements. This is an empirical gate, not a calibrated
upper confidence bound. The decision head is neither a failure probability nor
an expected-FDE regressor. The maximum of scores is a sign conjunction only;
its numerical magnitude has no risk interpretation.

Potential evidence would be a reproducible reduction of costly mistakes
without loss of utility or violation of the unchanged full-policy risk rules.
Lower fitting loss, an extra head, fewer interventions, or agreement between
two heads cannot establish this. Missing support remains NA, even when the
entire policy falls back. Independent scene confirmation is still required.

These controls do not reproduce the two published methods. A paper-level
comparison still needs task-matched published-method implementations with the
same predictor bank, source roles and action budget. No new theoretical
guarantee or submission-ready claim is supported by this experiment alone.
