# OpenDD: does the gap survive when both plans spend the same number of edits?

My earlier comparison gave the online rule and hindsight search the same
budget ceiling. That still let them spend different numbers of edits. Here I
fixed the online rule's actual count in each recording and seed and let the
hindsight solver choose where to spend those edits. Both use the same nested
candidate prefixes, chronological token rule (0.2 credit per eligible query,
two-credit cap), original 2% checks, and requirement to beat CV and OLS4.

| Comparison | Qualifying recording-seed views / 606 |
| --- | ---: |
| Frozen online rule | 57 |
| Hindsight, exact online edit count in each view | 242 |
| Hindsight, original quota ceiling | 330 |

The exact-count arms both spend 780 edits across 202 recordings, three seeds
and seven locations. Of the 88 quota-level witnesses lost when the count is
fixed, 80 are in views with no online edits and eight are at positive counts
where the quota-level witness cannot be recovered. Even after matching counts,
185 verified plans were missed by the online rule. The equal-location difference
is +17.41 percentage points (95% location bootstrap [+7.73, +28.84]; 3,000
draws), positive at six of seven locations. This interval describes sensitivity
across development locations, not a prospective improvement by a trained rule.

There are 267 zero-spend views. Of the 339 with positive counts, 242 qualify,
52 are risk-infeasible, 37 lack a strictly positive bound and eight lack support.
No solve remained unresolved. Matching counts and chronology does not match
total displacement across queries. The solver uses already exposed development
labels; it does not define an online policy or an external confirmation result.

## Local replay

From the repository root, with the frozen OpenDD checkpoints, development
cache, and original quota-level recording receipts:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_opendd_realized_stream_capacity.py
.venv-pytorch/bin/python scripts/report_m3w_opendd_realized_stream_capacity.py
```

The first command reruns the frozen controller with zero updates and verifies
its archived results by location and seed. It saves a private receipt per view
and resumes by checking existing receipts. The second command produces the
decomposition and paired location interval. See the [protocol](protocol.md),
[input registration](registration.json), [aggregate readout](readout.json)
and [decomposition](decomposition.json). No external label was reopened.
