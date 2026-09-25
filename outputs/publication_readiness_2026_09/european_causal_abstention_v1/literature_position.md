# Position Relative to Selective Prediction and Risk Control

## What Was Read

**SelectiveNet** (Geifman and El-Yaniv, ICML 2019): the primary PMLR paper,
including selective risk/coverage definitions, joint prediction/selection and
coverage calibration. Selective risk depends on which fraction is retained;
rejection can improve it without improving the underlying predictor. This motivates
our count-matched controls. We do not reproduce SelectiveNet: our abstention
retains a physical-motion forecast instead of omitting a prediction, and the
current study keeps the prediction and utility/risk heads frozen.
[Primary paper](https://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf).

**Conformal Risk Control** (Angelopoulos et al., ICLR 2024): primary arXiv PDF,
especially Section 1.1's exchangeable, bounded, non-increasing loss setup and
finite-sample correction. These conditions are not established here. Our opened,
overlapping source folds, learned H/B rule and rectangular feature support do not
provide that theorem's risk guarantee. Bootstrap intervals for locality-average
accuracy are not independent risk calibration. We do not call the proposed guards
conformal, distribution-free or physically safe.
[Primary paper](https://arxiv.org/pdf/2208.02814).

## Current Method Question

Can a controller estimate the incremental value and harm of replacing a competent
motion floor, abstain when current behavior lacks fitting support, and allocate
interventions among simultaneous agent queries better than count-matched controls?
The reference must be the forecast actually delivered on abstention, not a weaker
unprotected baseline. That distinction alone is not a novelty claim: the preceding
matched target experiment did not support changing the target reference.

This experiment tests an interpretable stopping/support filter under that fixed
system. Its same-recording/current-frame quota is an observational cohort allocation
comparison, not a learned interaction model or proof of reduced multi-agent conflicts.
Any gain must be separated from reduced intervention rate, retained-label bias,
lost useful interventions, source-producer transport and selection on opened data.

## What Is Still Missing

Neither architecture composition nor a convenient negative result establishes
a publishable contribution. A future main claim needs a matched, competitive
selective-regression baseline, explicit producer-chain independence, justified
calibration assumptions, actual interaction-consistency evaluation and untouched
confirmation scenes. This paper reading positions the experiment; it is not a
complete literature review or a novelty certificate.
