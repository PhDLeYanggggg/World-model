# Positioning The Amplitude Cost Experiment

Read-only primary-source refresh, 4 November 2026. This note does not report
new model performance, a novelty certificate or a theoretical risk guarantee.

[EqMotion, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_CVPR_2023_paper.html)
already learns equivariant motion features and invariant interaction reasoning.
Our frozen fixed-head adaptation is an existing proposal backbone. Its
performance is not an original M3W architecture contribution or a reproduction
of the published best-of-many prediction setting.

[Conformal Risk Control](https://arxiv.org/html/2208.02814v4) establishes expected
risk control under its bounded-loss, exchangeability and monotonicity setup.
It does not automatically certify every recording's baseline-relative error,
especially with an unavailable selected-reference denominator. Repeated
overlapping windows cannot simply replace exchangeable calibration units.

[Conformal Risk Control for Non-Monotonic Losses](https://arxiv.org/html/2602.20151v1)
extends the analysis through algorithmic stability and handles non-monotone
decision problems under stated conditions. Therefore non-monotonicity alone
is neither a new M3W problem nor an impossibility argument. This experiment
does not verify those theorem conditions or estimate a valid stability bound.

Our controlled empirical question is narrower: can correctly supervised
amplitude-conditioned costs recover some demonstrated action-class opportunity,
and does contemporaneous joint budget allocation help at the unchanged risk?
Plain convex interpolation, expected-cost regression and a mixed-integer
budget solver are not themselves claimed novel. The method contribution would
need independent empirical improvement and a justified account of selected
risk/support, beyond combining these existing pieces. Current empty-event
and original full-roster failures remain visible.

## A Mathematical Observation, Not A Guarantee

For a fixed input, label support and causal intervention direction, available
ADE along a straight interpolation path is convex in amplitude. Its positive
increase over the floor is convex, zero at amplitude0and nondecreasing for
nonnegative amplitudes. The easy-event indicator is amplitude-independent.
This is direct convexity algebra, not a new conformal theorem.

A single global shrinkage multiplier on a frozen intervention set could thus
give a monotone realized positive-harm path. Reoptimizing which agents/actions
are selected as the multiplier changes does not inherit that conclusion;
neither does it fix a zero reference denominator or establish exchangeability.
The current joint allocation experiment is not that fixed-set calibrated
procedure. No unimplemented monotonic calibration or population certificate
is attributed to it. Independent roles, suitable loss bounds and finite
support would still be required before any later formal claim.
