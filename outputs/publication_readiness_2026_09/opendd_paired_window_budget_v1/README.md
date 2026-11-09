# OpenDD: moving the same edit budget across neighboring windows

I kept the [retrained cost heads and forecast menu](../opendd_expanded_costs_v1/README.md)
and asked whether the original policy spends edits in the wrong windows.
Within each recording, I paired consecutive nonoverlapping windows. A
window left over at the end stays fixed. Each pair keeps the exact multiset
of intervention displacements; every window still has at most two edits
and at least one CV forecast. No slot is added, dropped, or enlarged.

For each pair, I enumerated the feasible slot placements. The additive and
joint decoders can search the same placements, using the same frozen
per-action gain and risk estimates. Both add the two window scores with
inverse roster-size weighting. Ties retain the original placement. The
fixed decoder reproduces all original paths element for element. Each
allocation is saved before outcomes are loaded. The pooled decoder sees
the later window's observed prefix, so this is a batch diagnostic, not
an online policy or an independent result.

| Arm | Exact edits | Original risk passes / 606 | Risk and both references / 606 |
| --- | ---: | ---: | ---: |
| Fixed additive | 3,690 | 38 | 32 |
| Fixed joint | 3,690 | 38 | 32 |
| Pooled additive | 3,690 | 31 | 27 |
| Pooled joint | 3,690 | 30 | 26 |

| Conservative contrast | Mean (pp) | 95% location bootstrap |
| --- | ---: | ---: |
| Pooled additive minus fixed additive | +0.04196 | [-0.19165, +0.29672] |
| Pooled joint minus fixed joint | +0.03791 | [-0.19462, +0.29866] |
| Pooled joint minus pooled additive | -0.01119 | [-0.03256, +0.00650] |

Across the 2,039 edited pairs, additive moves slots in 1,070 pairs and joint
in 1,040. Joint moves 727 slots earlier and 670 later. The small positive
mean for pooled versus fixed joint is sensitive to the four estimable views
at `rdb2`: excluding that location descriptively gives -0.07572 pp. The
primary intervals use all six estimable locations and 3,000 resamples.
The 606 views are 202 exposed recordings repeated across three seeds.

## Why risk passes fell

Both pooled arms lose 17 of their original 38 risk passes. Additive gains
10 new passes and joint gains nine. In 12 of each arm's 17 lost passes,
the selected easy reference becomes undefined even though the original
placement had a defined reference. This is a different failure from
exceeding the 2% rule. Of the lost passes, joint exceeds the all-risk
bound in five views and the easy-risk bound in four; reasons can overlap.
[Paired risk attribution](risk_attribution.json).

**Decision:** do not replace the fixed budget with this scheduler. Giving it
more freedom without preserving an easy reference removes evidence that the
original placement had. The next scheduler should explicitly preserve
predicted easy reference support, learn that estimate only on donor data,
and use only information available at the time of spending. Missing support
is still not a risk pass. No change to the original risk rule, reserved
outcomes, deployment, or working manuscript.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/audit_m3w_opendd_paired_window_budget.py freeze
.venv-pytorch/bin/python scripts/audit_m3w_opendd_paired_window_budget.py score
.venv-pytorch/bin/python scripts/audit_m3w_paired_budget_risk.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_paired_window_budget.py tests/test_m3w_exact_slot_costs.py tests/test_m3w_exact_slot_headroom.py tests/test_m3w_slate_shared_menu.py
```

The local arm64 run uses existing checkpoint weights and four Torch threads.
It adds no training. Predictions and aggregate scoring replay exactly.
[Registration](registration.json) | [Path freeze](freeze.json) | [Results](readout.json).
