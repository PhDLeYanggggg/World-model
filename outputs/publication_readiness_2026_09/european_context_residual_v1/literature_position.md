# Related-Method Boundary

## Primary Sources Inspected

Hebert-Johnson, Kim, Reingold and Rothblum, *Multicalibration:
Calibration for the (Computationally-Identifiable) Masses*, ICML 2018,
[paper](https://proceedings.mlr.press/v80/hebert-johnson18a.html).
Sections 2 and 3.1 define subgroup calibration and iterative corrections,
including the generalization problem posed by adaptive queries. This is not
a guarantee for our continuous trajectory-cost estimates under locality shift.

Kim, Ghorbani and Zou, *Multiaccuracy: Black-Box Post-Processing for Fairness
in Classification*, arXiv:1805.12317v2,
[paper](https://arxiv.org/abs/1805.12317).
Sections 2.3 and 3, including Algorithm 1, use residual-predicting auditors
and iterative post-processing; ridge and tree auditors are discussed.
The stated framework uses a representative audit distribution and controls
generalization, with fresh audit samples in the theoretical construction.
Its classification do-no-harm result does not certify our continuous cost,
overlapping windows, shifted localities or deployment policy.

## What This Experiment Does Not Establish

The fixed additive ridge probe is a diagnostic control, not a new
multicalibration or multiaccuracy algorithm. We do not inherit either
paper's guarantee. Our base predictions on residual-fitting rows are
in-sample; the probes are read on source-development held localities that
have been exposed in earlier experiments. Independent selection, calibration
and confirmation roles remain unopened.

The scientific question is narrower: does fitting-defined past-only context
improve easy-harm magnitude estimation on another locality beyond a global
intercept correction and the original strong cost estimator? A positive
diagnostic would still require a registered nested-OOF experiment, independent
risk calibration and a trajectory intervention comparison before a method
or deployment claim. A negative result cannot be repaired by relabeling this
probe as independent calibration.

Both sources were inspected on 2026-09-26. This note is method positioning,
not a systematic literature review or evidence of novelty.
