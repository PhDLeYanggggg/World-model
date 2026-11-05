# Method Positioning

This experiment tests an inductive bias for intervention-risk learning, not a
new motion architecture or an established novelty claim. Candidate predictions
are already causal outputs of frozen predictors. The new basis compares those
outputs with observed history before the risk head decides whether to admit them.
Future ground truth remains a supervised target, never a feature.

EqMotion already uses velocity magnitudes and velocity angles as invariant
pattern features, with geometric features and invariant interaction reasoning
(Sections4.1-4.3). Here the extra relations compare candidate and floor rollouts
against observed motion. They do not reproduce EqMotion's equivariant network;
signed lateral components are not reflection-invariant, and the complete input
network is not guaranteed rotation-invariant. Speed/angle features themselves
must not be claimed as a new contribution. [Xu et al., EqMotion, primary paper](https://arxiv.org/pdf/2303.10876).

Learn then Test separates a pretrained predictor from risk calibration using
valid hypothesis tests and family-wise error control (Sections1.1 and2.1).
Our fitting loss and exposed-development empirical checks do not satisfy that
theorem merely because a2% budget appears in the code. Repeatedly exposed
development is not fresh independent calibration, and empty support is not a
statistical safety pass. [Angelopoulos et al., Learn then Test, primary paper](https://arxiv.org/pdf/2110.01052).

The plausible research contribution remains jointly useful and harm-controlled
multi-agent intervention under source separation and incomplete outcomes. That
requires evidence beyond this feature experiment: original safety/support,
matched policy controls, independent recording-level uncertainty, and a fresh
confirmation evaluation after development is frozen. No theorem, world-dynamics
lift or submission readiness follows from a lower fitting MSE.

Primary-paper sections above were inspected on5November2026. The CVF HTML/PDF
route returned403 during the refresh; the authors' arXiv paper was available.
