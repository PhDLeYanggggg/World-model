# OpenDD: learning which equal-budget plan to prefer

The previous model estimated gain and risk separately for each edit count.
Here I trained directly on two plans for the same recording. They use the
same number of edits and have the same movement cap, but only one passes the
original risk and two-reference checks. I kept the query features, nested
forecasts and chronological credits fixed. Both new heads see the same pairs;
one learns a relative ordering, the other classifies plans independently.

| Offline allocator | Edits | Original risk passes / 606 | Passes risk and beats CV and OLS4 / 606 |
| --- | ---: | ---: | ---: |
| Frozen count scores | 780 | 88 | 60 |
| Independent pass classification | 780 | 86 | 54 |
| Pairwise preference | 780 | 95 | 59 |

At equal location weight, pairwise minus frozen is +0.032 percentage points
for the dual-pass rate (95% location bootstrap [-2.553, +2.419]). The paired
conservative gain is -0.02064 pp (95% [-0.04371, +0.00164]). Pairwise has five
more passes than classification in the pooled count; the equal-location
difference is -0.228 pp (95% [-2.751, +1.934]). The pooled difference does not
establish superiority at a new location. The pairwise head adds seven
original-risk passes versus frozen scores but does not add dual passes;
36 of its 95 risk passes lack a positive conservative gain over both
references, compared with 28 of 88 for the frozen scores.

| Held OpenDD location | Views | Frozen | Classification | Pairwise |
| --- | ---: | ---: | ---: | ---: |
| rdb1 | 381 | 48 | 38 | 42 |
| rdb2 | 6 | 0 | 0 | 0 |
| rdb3 | 15 | 3 | 3 | 2 |
| rdb4 | 21 | 0 | 0 | 1 |
| rdb5 | 45 | 1 | 2 | 1 |
| rdb6 | 135 | 6 | 9 | 11 |
| rdb7 | 3 | 2 | 2 | 2 |

Seven held locations and three seeds give 21 evaluation cells. Across their
fitting roles there were 6,842 verified pairs (159 to 408 per cell); those
are not independent recordings or unique across folds. Training used 42
heads and 21,000 updates, locally on native arm64, with checkpointing every
100 updates. The fresh run took 188.1 seconds. Checkpoint replay reproduced
the readout in 15.1 seconds with zero updates. One classification allocation
was unresolved and retained the exact frozen comparison plan; none of the
pairwise allocations was unresolved.

This isolates score ranking under fixed counts, not streaming admission.
The allocator uses the entire history-derived score sequence and the final
count. Adding a common reward per edit would not change a same-count
ranking, but it could change when a streaming policy edits. The run does
not identify that reward or a release threshold, and it does not warrant
changing the frozen policy. No external outcome was accessed.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/run_m3w_opendd_equal_spend_preference.py --replay
```

Omit `--replay` to fit or resume. The [protocol](protocol.md),
[registration](registration.json), [checkpoint manifest](head_freeze.json)
and [readout](readout.json) bind the split, training pairs and fitted outputs.
The earlier v4 anonymous archive does not include this later experiment.
