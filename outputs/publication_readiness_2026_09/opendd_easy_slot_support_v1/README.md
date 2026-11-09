# OpenDD: protecting easy reference support does not restore utility

The [paired-window scheduler](../opendd_paired_window_budget_v1/README.md)
often lost a risk pass because it stopped selecting agents with a usable
easy reference. I tested whether predicting that reference could prevent
the failure without increasing the edit budget.

I trained 21 small heads, one per held location and seed. Each sees motion
and neighboring agents from eight observed steps, and predicts the probability
of an observed, positive CV error below the easy cutoff and its conditional
amount. Their product estimates available easy reference mass. Training labels
and cutoffs exclude the evaluated location. A donor's own records do not set
its label cutoff. No held score chose the guard or its threshold.

For each pair of windows, keep the previously chosen pooled plan only if it
predicts at least as much total easy reference mass as the fixed plan.
Otherwise revert the entire pair. All plans keep the same 3,690 slots and
displacements; the 2% recording evaluator is unchanged. The predicted mass
is not a certificate. The test still uses two observed prefixes for allocation
and is not online deployment.

| Arm | Defined selected easy reference | Risk passes / 606 | Risk and both reference wins / 606 |
| --- | ---: | ---: | ---: |
| Fixed additive | not recomputed | 38 | 32 |
| Fixed joint | not recomputed | 38 | 32 |
| Pooled additive | 278 | 31 | 27 |
| Pooled joint | 282 | 30 | 26 |
| Guarded additive | 301 | 32 | 26 |
| Guarded joint | 305 | 33 | 27 |

The guard rejects 584 of 1,070 changed additive pairs and 579 of 1,040
changed joint pairs. Both arms add 23 views with a defined easy reference,
but this does not recover the original fixed-plan passes. Relative to
unguarded pooling, conservative utility changes by **-0.19278 pp**
[-0.35904, -0.06533] for additive and **-0.19142 pp** [-0.36096, -0.06364]
for joint. The intervals resample six estimable locations, not overlapping
windows or recording-seed repeats. The membership classifier averages
about 0.55 AUROC on those six locations; the seventh has no estimable
edit comparison and is not used to characterize scheduling performance.

**Decision:** do not deploy the guard. It makes the risk denominator more
often estimable, but preserving estimated support alone is not sufficient
to preserve accuracy or satisfy the original risk bound. Keep the fixed
reference as the comparison and do not use this result to justify another
large fit on the same development sources. The [per-site scores](readout.json)
retain the negative result and the support/utility trade-off. No reserved
outcomes, deployment or manuscript were changed.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/train_m3w_easy_slot_support.py freeze
.venv-pytorch/bin/python scripts/train_m3w_easy_slot_support.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_easy_slot_support.py tests/test_m3w_paired_window_budget.py tests/test_m3w_exact_slot_costs.py tests/test_m3w_easy_reference_rank.py
```

Native arm64 CPU, four Torch threads, zero workers. The runner saved loss,
optimizer state, random state and a PID heartbeat every 50 updates. The
21 checkpoint-based prediction and scoring replays were exact.
[Registration](registration.json) | [Frozen predictions](freeze.json).
