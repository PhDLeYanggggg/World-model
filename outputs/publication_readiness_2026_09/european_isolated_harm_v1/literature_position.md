# Relationship to Selective Prediction

SelectiveNet jointly trains prediction and rejection rather than adding only a
confidence threshold to a pretrained predictor. Its experiments evaluate the
risk-coverage tradeoff for classification and regression. Reference: Geifman and
El-Yaniv, ICML2019, PMLR97:2151-2159,
[original paper and abstract](https://proceedings.mlr.press/v97/geifman19a.html).

The current experiment is not an implementation of SelectiveNet. It estimates
benefit and positive harm relative to an existing motion floor, then applies
fixed all/easy reference-normalized risk conditions. It includes missing-outcome
completion bounds and recording transfer. Those choices require their own
evidence; a separate branch or reject option alone is not a novel contribution.

Freezing benefit/reference outputs while learning harm tests an architectural
tradeoff suggested by the preceding factorial. It does not directly measure
gradient conflict, prove causal mediation or establish calibrated population
safety. The experiment uses already exposed TRAIN-internal recordings. Any
positive mechanism result still needs independent protocol-compliant comparison
with published selective-prediction methods before a main scientific claim.

No independent confirmation or deployment promotion is recorded here. Primary
source checked2026-10-29; no third-party implementation or data was downloaded.
