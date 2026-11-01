# Method Positioning And Claim Boundaries

Fresh primary-source check:1November2026. This is a focused comparison,not a
systematic review or an empirical benchmark against these algorithms. The
current registered experiment is unchanged by this review.

## Multi-Expert Regression Deferral

Mao,Mohri and Zhong's [Regression with Multi-Expert Deferral](https://proceedings.mlr.press/v235/mao24d.html)
(ICML2024) studies both joint predictor/deferrer training and learning a deferrer
for pretrained predictors. Its framework includes multiple experts and bounded
regression costs,with consistency analyses for specified surrogates.

Selecting among fixed trajectory experts is therefore not a new problem merely
because the outputs are trajectories. The present actor is not that paper's
surrogate,and inherits no consistency result. A useful comparison would hold
forecasts,causal fields and data roles fixed while testing a compatible cost-aware
deferral baseline. Any bounded-loss conversion must be declared,not silently
clip large trajectory errors. Selected-query context,scene-query allocation and
the original recording-level risk contract require their own evidence.

Access note:publisher metadata/abstract verified;the linked PDF returned a web
challenge or unsupported content type. This is not a full-proof audit.

## Decision-Focused Learning

Elmachtoub and Grigas's [Smart Predict,then Optimize](https://arxiv.org/pdf/1710.08005)
defines decision loss from optimizing with predicted objective coefficients and
develops SPO+,a convex surrogate. The inspected introduction and contribution
statements describe linear-objective optimization,including mixed-integer feasible
sets,and give assumptions for consistency results.

Optimizing downstream decisions rather than moment MSE is thus established.
Our softmax/primal-dual actor does not implement SPO+,differentiate the MILP,or
inherit its guarantees. Inference additionally uses learned,frozen constraint
moments;calibration errors there are not cured just by learning objective scores.
The relevant question is whether the modified surrogate improves the actual
hard policy under unchanged risk criteria,not whether its training loss falls.

Access note:arXivv5PDF,19November2020,inspected introduction and contribution
sections. The publication and version are not being conflated with a new2026method.

## Conformal Risk Control

Angelopoulos et al.'s [Conformal Risk Control](https://arxiv.org/pdf/2208.02814)
has a result for expected risk under exchangeable,monotone loss functions and
a finite upper bound;the inspected Theorem1 also states regularity conditions.
The [ICLR2024 proceedings entry](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html)
verifies the venue. The inspected arXivPDF is a later2025revision.

An empirical2%budget or a locality bootstrap is not automatically conformal
control. Neither implies every recording satisfies a population safety guarantee.
Our adaptively exposed sources,overlapping windows,non-monotone action changes
and missing outcomes need explicit treatment before any theorem is invoked.
No current actor gate is described as a conformal or physical-safety certificate.

## Current Research Claim

The prospective distinction is a controlled separation of continuous forecast
costs,free seven-action routing,selected-query context and scene-level decisions
under partial labels and an unchanged recording-risk contract. This is a
hypothesis,not an established contribution. The minimum evidence still includes
a positive registered completion-lower contrast,matched component benefit,
all original risk views,and subsequent genuinely independent evaluation.
SOURCE fitting alone cannot establish cross-scene efficacy or submission readiness.
