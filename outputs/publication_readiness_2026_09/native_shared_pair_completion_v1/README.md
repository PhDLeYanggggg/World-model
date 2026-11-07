# Do Shared Targets Change The Pair-Geometry Conclusion?

Overlapping forecast windows can predict the same two people at the
same annotation step. Treating each missing occurrence independently
is valid but may charge incompatible worst cases to the same target.
I recalculated the bounds on the original Moving A pair predictions
while requiring such occurrences to have one target vector.

| Joint compared to | Original lower bound (px) | Shared-target lower bound (px), 95% CI | Gain in bound (px) |
| --- | ---: | ---: | ---: |
| Additive | -0.1580 | -0.1569 [-0.3359, -0.0362] | +0.0011 |
| Independent | -0.9445 | -0.9264 [-2.4351, -0.0438] | +0.0180 |
| Shared family | -1.2355 | -1.1998 [-2.8055, -0.1761] | +0.0356 |
| CV | -2.1672 | -2.1145 [-4.2683, -0.5065] | +0.0527 |
| OLS4 | -15.6363 | -14.5562 [-21.7169, -9.1479] | +1.0800 |

For joint versus additive the point completion envelope narrows
from [-0.1580, +0.1750] to [-0.1569, +0.1738] px. It still
spans both signs. The paired bootstrap interval for tightening
is [+0.000047, +0.002726] px; that is not an interval for
the actual unobserved pair effect. All five comparisons retain
their original inference conclusion: a conservative advantage
for joint selection is not established. In the primary comparison
877 nonzero missing occurrences share targets with others, and
134 recording/pair/step/seed groups admit a tighter plan.

## How The Bound Works

Within a recording and seed, let two forecast sets for the same
unknown relative position be `a_i` and `b_i`, weighted by their
original inverse query denominators `w_i`. Any coupling with both
marginals `w` rewrites the weighted contrast as a sum of paired
differences of distances to the same unknown point. The reverse
triangle inequality bounds its absolute value by the coupling's
Euclidean transport cost. See [Peyre and Cuturi, Computational
Optimal Transport](https://optimaltransport.github.io/pdf/ComputationalOT.pdf)
for the transport construction. The diagonal coupling gives the
original bound. We use exact assignment for equal weights and a
feasible mass-preserving swap plan otherwise; the latter is not
claimed optimal. No missing location is imputed.

The key uses recording, ordered agent IDs and absolute annotation
step. All 1,323,409 adjacent duplicate agent-step checks agree in
both released masks and available positions. That checks within-recording
identity consistency, not independence across datasets. We keep
all 34 exposed recordings, 1,613 queries repeated across three
seeds, and 9,675 switches per matched arm. The original bounds and
source-bootstrap endpoints reproduce to numerical tolerance.
The 3,000 paired bootstrap draws are over 14 sources after seed
averaging. At a fixed terminal lead there is no repeated target
across origins within a recording, so FDE is unaffected.

This is an 8/12-step image-pixel sensitivity only. The official
2% recording rule, individual utility, forecasts, selection, and
external protocol are unchanged. No independent result is claimed.

~~~sh
.venv-pytorch/bin/python scripts/audit_m3w_shared_pair_completion.py score
.venv-pytorch/bin/python scripts/audit_m3w_shared_pair_completion.py replay
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_shared_pair_completion.py \
  tests/test_m3w_joint_path_audit.py
~~~

Six focused tests passed. The fresh score and full replay each
took approximately 20 seconds after imports and produced the
same aggregate hash.
[Protocol](protocol.md) | [Analysis](analysis.json) | [Replay](replay_execution.json).
