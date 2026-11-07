# Where The Fixed-Bank Intervention Capacity Is Lost

Fresh hindsight diagnostic on the full exposed Moving A population:
125,540 rows, 1,613 queries, 34 recordings, 14 sources, and 102 repeated
recording views across three existing forecaster seeds. Eight observed
and twelve future annotation steps, in native image coordinates.
Candidate paths and the direct arm are frozen. All arms retain the
original 2% and reference-support conditions, including the signed easy
limit. Hindsight uses development targets, so this is not a policy.

| Nested problem | Checked feasible | Proven infeasible | Unresolved |
| --- | ---: | ---: | ---: |
| Free counts within the original per-query cap | 101 | 0 | 1 |
| Direct count fixed in every query | 81 | 20 | 1 |
| Fixed counts plus predicted totals at recording end | 62 | 36 | 4 |
| Fixed counts plus predicted totals at every prefix | 48 | 40 | 14 |

Every checked witness has a positive paired completion-lower gain
versus the direct control and positive lower gain versus CV. The
first two rows' witnesses are solver-optimal; the terminal and
prefix rows include 7 and 12 feasible witnesses without optimality
proof. The last row has 12 solver-limited views and 2 numerical
incumbents rejected by the unchanged postcheck. No incomplete
subset is reported as a whole-roster utility estimate.

Fixing the old switch counts blocks 20 previously feasible views:
18 with zero switches and 2 with no eligible easy reference. Adding
only the terminal predicted budgets blocks another 16. Enforcing
every prefix blocks four more, and turns 12 previously feasible
views into unresolved ones. This distinguishes estimated aggregate
cost restrictions from the additional chronological constraint.
It does not establish that those 12 are infeasible.

The free-count witnesses use 9,391 switches across their feasible
views; they are not a matched-volume performance control for the
old 6,952. The evidence supports available useful trajectories on
this development bank. It does not show that a causal allocator can
identify them, or isolate redistribution from switch volume.
Next, match the total per recording while allowing timing to move;
then test a past-only allocation on the original risk evaluator.

Runtime: the original solver's dominated-column presolve exceeded
the requested limit. After two sampled and explicitly stopped
processes, the final solver bypassed that presolve and reused 26
completed views (20 and 6) rather than restarting them. The final
run completes the rest in 357.4 seconds; aggregation replay matches
its analysis byte-for-byte without re-solving. No choices are
exported, no independent labels are accessed, and deployment,
Stage5C and SMC are unchanged.

[Aggregate results](summary.json) | [Reproduction and recovery](operation_zh.md)
