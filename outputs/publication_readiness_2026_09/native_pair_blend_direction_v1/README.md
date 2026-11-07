# Query-Local Pair Headroom Does Not Establish Recording Feasibility

This hindsight diagnostic keeps the same 9,675 original switch
recipients, query counts, individual and pair-trained predictions
and CV fallback. It varies only the continuous blend weights of
those recipients. Each positive witness is independently re-scored
with the original pair evaluator and remains a genuine switch.

| Weight constraint | Verified gain | No gain to numerical tolerance | Unresolved |
| --- | ---: | ---: | ---: |
| One shared weight per query | 2,746 | 2,093 | 0 |
| Separate weights, jointly evaluated | 3,598 | 1,241 | 0 |

The 4,839 active views repeat 1,613 queries across three seeds. They
are not independent observations. Separate weights add 852 verified
views on which the shared-direction bound rules out a gain to
tolerance. The source-balanced verified rates are 52.08% [38.59,
64.77] and 69.61% [56.38, 81.32]. The mean witnessed lower gains
per active query are 0.0206 px [0.0142, 0.0280] and 0.0301 px
[0.0189, 0.0443]. These 95% intervals average seeds first and
bootstrap the 14 sources in 3,000 paired draws.

Assembling the witnesses into complete recordings does not increase
original risk passes: the individual reference, shared and separate
witnesses pass 5/5/5 of 102 views. The separate arm improves
conservative individual utility relative to the individual reference
by +0.037 pp [+0.013, +0.064], but its comparison against OLS4 is
-8.976 pp [-20.582, -0.135]. Worst signed easy degradation remains
65.52%. The witnesses were optimized per query for pair gain, not
jointly for recording feasibility. The result establishes headroom
in this affine family, not that no recording-feasible blend exists.

## Directional Test

For pair-step residual $e$ and affine change $D\alpha$, the
unnormalized loss increment is the sum of
$\lVert e+D\alpha\rVert-\lVert e\rVert$ on co-observed steps
and $\lVert D\alpha\rVert$ on unobserved steps. The full requested
edge-step count is the denominator. Let this normalized convex
function be $F$, with $F(0)=0$. Every nonzero feasible blend
can be expressed as $t u$ with $t>0$, $u\geq0$, and
$\sum_i u_i=1$. Convexity gives $F'(0;u)\leq F(tu)/t$;
a negative derivative also implies a negative increment at
sufficiently small positive $t$. The starting paths are
nonidentity, so such a small step does not change the count.

On co-observed steps with nonzero residual the derivative is
linear in $u$. On the remaining steps it is a sum of norms.
One-recipient directions are exact; with two recipients, convex
subgradients bracket the minimum on one interval. A bound of
at least -1e-9 px per unit blend sum rules out more than 2e-9
px improvement for the corresponding query. Positive results
come from finite, re-scored ray witnesses; their magnitudes
are not global maxima over all blend weights.

[Protocol](protocol.md) | [Registration](registration.json) |
[Analysis](analysis.json) | [Score](score_execution.json) |
[Replay](replay_execution.json).

```sh
.venv-pytorch/bin/python scripts/audit_m3w_pair_blend_direction.py score
.venv-pytorch/bin/python scripts/audit_m3w_pair_blend_direction.py replay
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_pair_blend_direction.py \
  tests/test_m3w_completion_motion_blend.py tests/test_m3w_joint_path_audit.py \
  tests/test_m3w_dual_reference_capacity.py
```

Eleven focused tests passed; the full directional scan and
recording scores replay exactly. Exposed development only;
no oracle choices, forecasts or labels are published. Units
are native image pixels and eight/twelve annotation steps.
No external access, deployment, Stage5C execution or SMC.
