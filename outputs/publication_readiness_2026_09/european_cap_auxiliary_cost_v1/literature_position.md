# Evidence and Literature Position

## Probability Is Not Expected Cost
Binary logarithmic and Brier scores assess predictive probabilities. Proper
scoring rules reward the data-generating distribution in expectation; this
does not mean any finite fitted network is calibrated or transferable.
See Gneiting and Raftery (2007), Section 3, Examples 1 and 3 and Table 1,
[Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).

For this experiment, let E be the producer-relative cap-exceedance event and
Y the realized easy-harm cost. The elementary conditional-mean decomposition
is

    E[Y | X] = P(E | X) E[Y | X,E] + P(not E | X) E[Y | X,not E].

This is an explanation of our estimands, not a new theoretical contribution.
Learning the first probability alone need not recover either conditional
magnitude. Further, a shared finite encoder optimized with two losses can
trade off their errors. Therefore we measure cost accuracy directly instead
of promoting the preceding event-classification result as cost improvement.

The auxiliary experiment is ordinary shared-representation multitask learning
with a problem-specific label and matched controls. The architecture or the
addition of binary cross entropy alone is not claimed as methodological
novelty. A positive cost result would be one supporting component of the
broader forecasting-intervention question, not proof of that contribution.

## Cost Accuracy Is Not Risk Control
Learn then Test separates a learned predictor from calibration and formulates
risk control using hypothesis tests. Its stated calibration setup uses i.i.d.
observations, a specified risk, and tolerance/error levels fixed in advance;
it can abstain when no setting is certified. See Section 1.1 and Section 2 of
[Learn then Test](https://arxiv.org/html/2110.01052v5).

Our source-development locality bootstrap is not that procedure. Repeated
windows, three seeds and six overlapping assignments do not create extra
independent calibration scenes. A favorable MSE interval, output nesting or
mean predicted/actual cost ratio is not a finite-sample deployment guarantee.
Independent calibration and confirmation remain unopened. No LTT guarantee
is imported into this experiment, and no policy threshold is selected here.

## What Would Still Be Required
Beyond a positive matched cost comparison, a paper claim needs fixed-
forecaster, matched-intervention comparisons, scene-joint utility, applicable
risk calibration and independent confirmation. If this study is negative,
retain the result and diagnose the estimand and transfer error before further
model search. Neither a favorable event classifier nor a lower fitting loss
can substitute for those missing experiments.

Sources were opened on 2026-09-26. This note adds interpretation, not a change
to the committed protocol, loss, primary metric or acceptance rule.
Obs8/pred12 native annotation steps and detector-image pixels only. Stage5C
and SMC remain off; no physical-safety, metric, seconds-level, true-3D or
foundation-model claim.
