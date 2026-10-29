# Method Positioning

This experiment changes a trajectory regression loss; it does not establish a
new theory of selective prediction or a novel model merely by adding penalties.

[SelectiveNet (Geifman and El-Yaniv, ICML2019)](https://proceedings.mlr.press/v97/geifman19a.html)
jointly learns prediction and rejection for a risk-coverage tradeoff. Our present
experiment does not implement its joint reject head: it repairs a candidate
forecaster before fitting a new intervention policy. Full policy performance
relative to an active motion fallback remains a separate question.

[A Reductions Approach to Fair Classification (Agarwal et al., ICML2018)](https://proceedings.mlr.press/v80/agarwal18a.html)
reduces constrained classification to cost-sensitive problems. It motivates
careful loss/constraint accounting, but its classification guarantees do not
automatically apply to our dependent trajectory windows, incomplete futures or
nonconvex neural optimizer. We make no such guarantee.

The prospective research contribution must instead be demonstrated through
reliable relative gain/harm learning, source-excluded candidate support, and
joint multi-agent intervention compared at equal intervention count and fixed
scene-level budgets. A positive development loss comparison alone is insufficient.
Independent risk calibration, unused-scene confirmation and strong public-method
comparisons are still required. This note is an abstract-level primary-source
positioning check, not a completed literature review or novelty verdict.
