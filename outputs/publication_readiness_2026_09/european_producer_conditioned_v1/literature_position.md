# Relation To Stacking

Primary source checked 2026-09-25: Leo Breiman, *Stacked Regressions*, Machine
Learning 24, 49-64 (1996), especially Sections 1 and 3.
[University-hosted original paper](https://statistics.berkeley.edu/sites/default/files/tech-reports/367.pdf).

Breiman describes learning a combination from out-of-fold predictions rather
than from predictions on the same observations used to fit the base models.
His tree experiment learns combination weights from fold-specific models and
then evaluates a predictor built from trees fitted on the full fitting set.
This distinction makes the identity and training exposure of the forecast
producer relevant to our audit. It does not establish that this transition
causes our observed errors or that a producer tag solves them.

Our study instead learns bounded benefit/harm moments for choosing between a
causal fallback and a neural trajectory. The newly tested input is the identity
of an already-trained, source-excluded producer bundle. Global, real-tag and
placebo-tag heads use identical forecast pairs. This is a falsifiable diagnostic
extension, not evidence that stacking or model-conditioned gating is new.

The historical producer-substitution experiment already failed. It must not be
repackaged as a successful new method. Any eventual contribution needs positive
identical-forecast controls, robust source-level uncertainty, an independently
calibrated risk claim and an untouched confirmation evaluation. This development
experiment alone does not supply those conditions.
