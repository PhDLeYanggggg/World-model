# Primary-Source Update

Checked 5 October 2026. This is a targeted comparison, not an exhaustive
systematic review. Earlier sealed reports are not rewritten.

## Previously Unresolved Reference

The September source-intervention note could not assess *Conformal Policy
Control* through OpenReview. Its
[arXiv v1 full text](https://arxiv.org/html/2603.02196v1) is now accessible and was read through
Sections 4.1-4.2. Theorem 4.5 uses exact weights and states a stability-dependent
bound, not an unconditional certificate. Practical weight approximations are
distinguished in the paper. Our deterministic selector has neither the required
policy likelihood construction nor demonstrated calibration assumptions.
Therefore generic safe-reference regulation is prior work, and no guarantee
transfers to our selected-harm ratio. Its empirical reproduction remains not_run.

## Consequence for M3W

We should not present the architecture combination, a fallback switch or a risk
threshold as sufficient novelty. The current empirical question is whether
temporal cost supervision improves controlled intervention utility while
preserving selected harm against strong simple estimators. The new manuscript
states this as unresolved and reports the negative evidence that motivates it.
This positioning changes no experiment, threshold or role assignment.

Other primary sources revisited:

- [Regression with Multi-Expert Deferral](https://proceedings.mlr.press/v235/mao24d.html): pretrained-predictor routing is already an explicit problem setting.
- [Learn then Test](https://arxiv.org/abs/2110.01052v5): calibration is an inferential procedure, not merely choosing a threshold.
- [Conformal Risk Control](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf): its theorem's assumptions have not been established for our selected-risk rule.
- [EqMotion](https://arxiv.org/abs/2303.10876v2) and [JFP](https://arxiv.org/abs/2212.08710): predictor and joint-interaction precedents, not methods invented by this project.

This comparison identifies overlap and missing assumptions. It neither proves
our method is novel nor proves a competing method will solve the present task.
