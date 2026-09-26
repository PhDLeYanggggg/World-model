# Method Position and Claim Boundary

Primary-source reading refreshed 2026-09-26 while the preregistered cost heads
were running. This note changes no model, endpoint, threshold or readout rule.

## Regression Deferral Is Prior Work

Mao, Mohri and Zhong study regression with multiple experts, including a
two-stage setting with a fixed predictor and a learned deferral function.
Their framework accommodates bounded regression losses and instance- or
label-dependent costs, with consistency results for their surrogate losses.
Our cost-aware switching motivation is therefore not itself a new problem.
The present conditional-mean decomposition does not inherit their theorem.
Reading scope: abstract, Section 2 definitions and Section 4 two-stage setup.
[Regression with Multi-Expert Deferral, ICML 2024](https://proceedings.mlr.press/v235/mao24d.html),
[author-hosted paper](https://cs.nyu.edu/~mohri/pub/regdef.pdf).

## Risk Guarantees Need Their Assumptions

Angelopoulos et al. give risk control in expectation for exchangeable,
bounded, non-increasing and right-continuous loss functions with an
appropriate finite-sample correction. They discuss the failure of the basic
procedure for non-monotone losses. Our locality bootstrap is an exploratory
uncertainty analysis, not that calibration algorithm. Neither repeated source
roles nor a bounded neural output establish its assumptions. Reading scope:
Section 1.1, Theorem 1 and Section 2.3.
[Conformal Risk Control, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf).

## What This Experiment Can Establish

The identity E[H 1(E)|x] = P(E|x) E[H|E,x] is ordinary conditional expectation,
not a proposed theoretical contribution. Here E means the fitting-defined
easy event; H is relative positive prediction error, not physical collision
harm. Fitting a membership classifier and conditional cost regressors is a
component hypothesis. Its value must survive matched controls and locality
transfer, including the original cost model and constant-probability control.

Even a positive cost result would not establish improved trajectories,
scene-joint allocation, calibrated deployment, learned physical dynamics or
multimodal world-model capability. The eventual method claim would require
fixed-forecast policy comparisons under the same intervention/risk budget,
independent scene evaluation and a separately justified calibration rule.
This round performs none of those policy or independent-data experiments.

Pixels and annotation steps remain the units. Detector labels are not human
gold. No true3D, foundation, metric, seconds-level or submission-ready claim;
Stage5C and SMC stay disabled.
