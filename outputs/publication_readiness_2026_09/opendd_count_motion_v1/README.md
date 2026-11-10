# OpenDD: more movement is not what closes the gap

I kept each recording's actual edit count, the nested forecast menu and the
chronological credits fixed. I also capped the sum of the edited agents' peak
displacements from CV at what the frozen controller spent in that recording
and seed. An alternative can spend less, not more. This is a forecast-change
budget in the dataset's coordinate units, not a physical safety measure.

| Allocation on the same 606 recording-seed views | Beats both references and passes original risk |
| --- | ---: |
| Frozen online controller | 57 |
| Offline reordering of frozen logits, no labels, count and movement matched | 60 |
| Label-aware search, count and movement matched | 210 |
| Label-aware search, count matched only | 242 |

Every arm makes 780 edits, with identical per-view counts. The movement-capped
hindsight gap is +14.68 percentage points at equal location weight (95%
location bootstrap [+5.69, +24.44], 3,000 draws). The gap is positive in five
of seven locations; leave-one-location-out estimates are +10.71 to +17.13 pp.
The stricter budget removes 32 of the 242 earlier *verified* cases; two of
those comparisons may still be feasible but did not pass the numerical check.
There were 267 views with no edits, which are not positive witnesses.

The score-only reallocation changes 73 plans, gains eight passes and loses
five. Its equal-location change is +0.39 pp (95% interval [0.00, +1.03]). It
can see the complete sequence of scores and the final count, so it is an
offline control, not a causal deployment result. The gap left by this control
suggests that changing the scheduler alone is not enough for this frozen
score. It does not show that a different learned score would close the gap.

## Reproduction

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_opendd_count_motion.py
```

Use the same local checkpoints and [fixed-count receipts](../opendd_realized_stream_capacity_v1/README.md).
The run resumes by checking each saved plan and replaying the original
scorer. The first numerical trial stopped after rejecting a purported
optimum that lost to an explicitly feasible online plan. The final run
uses no MILP presolve and tests that boundary case against exact constraints.
There were zero model updates and no new access to confirmation outcomes.
See the [protocol](protocol.md), [registration](registration.json) and
[readout](readout.json).
