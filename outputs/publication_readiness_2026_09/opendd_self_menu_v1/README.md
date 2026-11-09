# OpenDD: a better forecast menu does not fix the frozen selector

I added a [self-history neural forecast](../opendd_velocity_candidate_v1/README.md)
to the original OLS4 and median-slope menu. Its checkpoint excludes the scored
location and sees only the eight observed positions of the target agent.
Both additive and joint selection can use it. I kept the original selector
weights, eligible agents, and all 3,690 edit slots with their exact peak
displacements. A candidate can be shortened to a slot, not enlarged. The
old menu reproduces the original paths element for element.

| Selector | Menu | Risk passes / 606 | Risk and positive gain against CV and OLS4 / 606 |
| --- | --- | ---: | ---: |
| Additive | Original | 39 | 32 |
| Joint | Original | 41 | 34 |
| Additive | Plus self-history | 35 | 30 |
| Joint | Plus self-history | 40 | 34 |

| Conservative comparison | Site mean (pp) | 95% paired location bootstrap |
| --- | ---: | ---: |
| Expanded vs original, additive | +0.02584 | [-0.01463, +0.07326] |
| Expanded vs original, joint | +0.01581 | [-0.02619, +0.05324] |
| Joint vs additive, expanded | -0.02835 | [-0.06526, +0.00230] |

Additive and joint select 1,092 and 836 edits, respectively, that only the
neural path can provide. The new candidate is used, but it does not produce
a reliable gain or an increase in dual passes. The interval resamples six
estimable locations 3,000 times after averaging comparable recording-seed
views within each location. The 606 views are 202 recordings across three
existing trained seeds, not independent samples.

## Does the expanded menu contain usable alternatives?

I repeated the [exact-slot hindsight check](../opendd_exact_slot_headroom_v2/README.md)
with the additional forecast. The base additive policy remains the same
reference for both oracle arms. Single-edit choices are held fixed. Only the
two-edit assignments can change, and the original missing-future and 2% risk
checks still apply.

| Hindsight menu | Risk and positive gain against CV and OLS4 | Also beats base additive |
| --- | ---: | ---: |
| Original | 38 / 439 | 9 / 439 |
| Plus self-history | 50 / 439 | 25 / 439 |

There are 16 newly positive views and no unresolved solves. The 25 are from
15 recordings at three locations. This shows the larger menu contains more
feasible choices, not that a learned selector can find them or that it beats
an additive selector trained on the expanded menu.
[Capacity results](../opendd_self_menu_capacity_v1/readout.json).

**Next experiment:** train the gain and harm scores on the expanded menu
with forecasts that exclude both the donor and outer location. Compare the
retrained joint and additive arms at the same budget. Do not add another
candidate or change the risk rule before testing that mismatch. These are
exposed OpenDD development results. No new model fit, reserved outcome
access, deployment change, or manuscript edit occurred.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/audit_m3w_opendd_self_menu.py freeze
.venv-pytorch/bin/python scripts/audit_m3w_opendd_self_menu.py score
.venv-pytorch/bin/python scripts/audit_m3w_opendd_self_menu_capacity.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_opendd_self_menu.py tests/test_m3w_exact_slot_headroom.py tests/test_m3w_slate_shared_menu.py tests/test_m3w_pair_effect.py
```

The local arm64 CPU run uses four Torch threads and no workers. The per-fold
forecasts and aggregates replay exactly. A fresh solver run in an empty
cache also reproduced every capacity checkpoint hash.
[Registration](registration.json) | [Forecast freeze](freeze.json) |
[Causal results](readout.json).
