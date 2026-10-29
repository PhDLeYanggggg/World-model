# Method Position and Missing Published Controls

Primary sources checked on 29 October 2026 while the registered fits were
running. This note does not change the frozen experiment or its thresholds.

## Multi-Expert Regression Is Established Work

Mao, Mohri and Zhong, *Regression with Multi-Expert Deferral*, ICML 2024,
already study both joint predictor/deferral learning and a fixed-predictor,
learned-deferral setting, with continuous losses and multiple experts. Their
Section 4 is particularly relevant to our frozen action bank. Therefore,
choosing a motion or neural forecast through learned continuous costs is not
by itself a new contribution. Their surrogate is a required published control,
not an algorithm implemented by the current five-moment regressor.
[Proceedings](https://proceedings.mlr.press/v235/mao24d.html),
[author-hosted paper, Section 4](https://cs.nyu.edu/~mohri/pub/regdef.pdf).

Cortes et al., *Optimized Deferral for Imbalanced Settings*, ICML 2026, address
expert imbalance through a cost-sensitive formulation and margin-based losses.
It is directly relevant if a learned bank collapses toward a dominant baseline
despite a favorable oracle. The current paired experiment can diagnose that
failure but does not reproduce their MILD method or inherit its guarantees.
[Original proceedings and abstract](https://proceedings.mlr.press/v306/cortes26b.html).

## A Predicted 2% Condition Is Not Conformal Safety

Angelopoulos et al., *Conformal Risk Control*, ICLR 2024, Theorem 1, assumes
exchangeable, bounded, non-increasing loss functions and proves an expected
risk statement under its calibration rule. Those assumptions cannot be
replaced by a fitting MSE or a confidence threshold. Our recording-wide
positive-harm/reference ratios are not automatically monotone as selected
sets shrink, and an expected future-risk statement is not a guarantee for
every recording. [Original paper, Sections 1-2](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf).

A direct mathematical counterexample is enough to show the ratio problem.
Two selected actions each have reference cost 1, with harms 0 and 0.03.
Their combined ratio is 0.015. Raising a cutoff that removes only the first
action leaves 0.03. Thus making fewer interventions need not lower the ratio.
This is our elementary observation, not a new theorem attributed to that paper.
Empty selected support also remains undefined under our approved protocol.

## What Must Be Demonstrated for This Project

The prospective experiment tests causal learnability and incremental neural
value, not novelty of multi-expert routing. A paper-level contribution would
need to show that explicit baseline-relative benefit/harm, missing-outcome
completion bounds and recording-level evaluation change a useful decision,
then compare independent, scene-uniform and scene-joint policies under matched
predictors and intervention counts. Published deferral and calibrated-selection
controls, independent recording calibration/confirmation, and interaction
consistency remain required evidence. Neither an oracle bank nor a positive
development contrast can substitute for them.

No theorem of consistency, conformal validity, population safety or independent
generalization is claimed for the current MLP, fixed harm weights or threshold.
