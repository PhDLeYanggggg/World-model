# Which Missing Positions Are Worth Reviewing First?

I fixed the joint and additive forecasts and asked how many missing
coordinates would have to become available to reduce the original
pair disagreement debit. The plan reads availability masks and
forecast disagreement, not missing or observed target coordinates.
It uses the 34 exposed recordings, 14 equally weighted sources and
the same three seed forecasts.

| Original debit removed | Chronological | Incident mass | Dependency-aware | Fractional lower bound |
| ---: | ---: | ---: | ---: | ---: |
| 50% | 2,755 | 361 | 332 | 310.79 |
| 75% | 3,280 | 786 | 689 | 678.24 |
| 90% | 3,386 | 1,337 | 1,287 | 1,264.06 |
| 95% | 3,454 | 1,706 | 1,691 | 1,665.69 |
| 99% | 3,488 | 2,324 | 2,322 | 2,317.25 |

The entries count unique recording-agent-steps. Of 40,134 missing
positions, 3,492 affect this fixed comparison. The chronology
control scans only those 3,492, not the whole roster. At 95%,
prioritizing by forecast disagreement does most of the work:
explicit dependency tracking saves a further 15 positions versus
incident ranking. The original debit is 0.16651 image pixels,
and the four largest recording contributors account for about
69.8% of it.

The pair grid yields 7,074 changed missing occurrences before
aggregation, or 3,695 distinct dependency terms. A singleton term
is resolved by one position; a double term needs both. At each
step, the dependency-aware rule requests the singleton or linked
pair with the greatest newly resolved mass per position. The
continuous relaxation lower-bounds the optimal integer budget
at the same removal level; the greedy numbers are feasible
prefix counts, not claimed exact optima.

**Interpretation:** These are conditional review budgets. A truly
out-of-frame point may not be annotatable. Valid new coordinates
could shift the measured effect either way. The 95% figure is
for the original diagonal disagreement debit, not a guaranteed
shrinkage of the tighter shared-target envelope, confidence
interval, or guarantee of statistical resolution. No annotation
has been collected. This result does not change the official
metric, original 2% gate, or external confirmation policy.

~~~sh
.venv-pytorch/bin/python scripts/audit_m3w_pair_annotation_budget.py score
.venv-pytorch/bin/python scripts/audit_m3w_pair_annotation_budget.py replay
.venv-pytorch/bin/python scripts/plot_m3w_pair_annotation_budget.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_pair_annotation_budget.py \
  tests/test_m3w_pair_completion_witness.py tests/test_m3w_anchored_pair_completion.py \
  tests/test_m3w_shared_pair_completion.py
~~~

Nine focused tests passed. Fresh score and full replay agree
exactly; each took about 2.7 s after imports. The local figure
is generated from the verified aggregate, with PDF SHA-256
`12fcd2cd6d2ba0a656cb40d218188fca111c8a30100c5fa16610c9507d83a2a4`.
No annotation identities, trajectory arrays, external labels,
deployment, Stage5C or SMC.
[Protocol](protocol.md) | [Analysis](analysis.json) | [Replay](replay_execution.json).
