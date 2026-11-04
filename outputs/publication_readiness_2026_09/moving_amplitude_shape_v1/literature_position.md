# What A Shape-Constrained Follow-Up Could Claim

Primary-source check, 4 November 2026; scope is the original official
abstracts and introductory discussion, not a comprehensive novelty review.

[Input Convex Neural Networks, ICML 2017](https://proceedings.mlr.press/v70/amos17b.html)
already constrains a neural output to be convex in selected inputs, while
allowing other context dependence. Convexity of an output is not convexity
of neural training and does not guarantee accurate costs or generalization.

[Deep Lattice Networks and Partial Monotonic Functions, NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)
already learns partially monotone functions using constrained layers and
piecewise-linear calibration. A monotone or convex readout is therefore
not itself an original architecture contribution here.

Our direct algebra is specific but elementary: with a fixed valid-label mask,
available ADE along floor+alpha*(proposal-floor) is convex. Positive harm is
its positive increase, hence convex, nonnegative, zero at alpha0 and
nondecreasing. The amplitude-independent easy indicator preserves this.
Both easy and non-easy positive-harm contributions are convex, so their
conditional expectations given the same causal history retain that shape.
No outcome information is required to parameterize a function with this shape.

A nonnegative normalized hinge mixture is one possible parameterization, not
an implemented or effective repair merely because this note describes it.
It would need a prospectively fixed matched training control and unchanged
recording-risk evaluation. Shape consistency cannot repair a zero selected
reference denominator, missing labels, inaccurate magnitudes or unverified
source exchangeability. No CRC guarantee, new trajectory backbone, independent
evidence, deployment or submission readiness follows from this diagnosis.
