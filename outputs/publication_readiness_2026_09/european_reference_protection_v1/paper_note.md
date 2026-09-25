# Paper Evidence Note

## Question
When most trajectories are adequately served by a causal reference forecast,
can source-trained cost estimates allocate neural intervention across a scene
without transferring excessive error to easy cases?

## What This Experiment Adds
This is a controlled negative result about a plausible repair. It separates
extra optimization budget from reference-cost preservation, holding forecasts,
source roles, sample sequence and loss scales fixed. Freezing reference moments
does not yield a statistically clear all-ADE advantage over matched continuation
and does not resolve observed harm violations. Joint allocation continues to
show an accuracy effect relative to independent gates, but it spends more
realized risk. These two findings must be stated together.

## Relationship to Prior Work
[Yu et al., Gradient Surgery for Multi-Task Learning, NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)
motivates investigating interference between tasks. This study does not measure
their proposed sufficient conditions, implement PCGrad or reproduce their
experiments. Frozen branches alone are not an architectural novelty claim.
Our scientific question is transported relative harm under scene-query
selection; a contribution there still needs evidence beyond this diagnosis.

## Appropriate and Inappropriate Claims
Appropriate: a matched source-development experiment did not validate the
reference-protection repair; mean easy preservation and positive-harm control
give materially different assessments; query allocation has an accuracy/risk
tradeoff worth studying with stronger controls.

Inappropriate: safe deployment, independent confirmation, new world dynamics,
metric or seconds-level prediction, true3D/foundation success, or submission
readiness. No claim is rescued by choosing the best seed, source assignment
or secondary gate after reading C outcomes. The result is evidence for method
development and limitations, not yet a main positive paper result.

## Reproducibility Scope
The package includes registered configuration, decision freeze, model/data
cards, training losses, all-source comparisons, B/C fitting diagnostics and
scoped regression/replay records. Private checkpoints and raw data are not
published. The final verification records exactly which tests and replays ran;
it is not a claim that the entire historical repository test suite passed.
