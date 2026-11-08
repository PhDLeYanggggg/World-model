# When to Spend a Support Edit

I held the first gain edit, the 345 eligible second-slot views, the
0.10 support amplitude and the easy-score threshold of 0.5 fixed.
At the first eligible query a support credit becomes available.
The policy can spend it there or within the next two queries. It
can expire unused, and the end of the recording does not force release.
The learned heads see only current features, query ordinal and credit
age. The training-only continuation target contrasts immediate
risk-adjusted gain with the best later training opportunity or abstention.

The 606 recording-seed views cover 202 previously exposed recordings
across seven OpenDD sites and three seeds. For each evaluation site and
seed, forecast teachers exclude both the evaluation site and the donor
site. These are development comparisons, not independent confirmation.

| Policy | Support edits (delayed) | Undefined risk | Risk pass | Joint pass |
| --- | ---: | ---: | ---: | ---: |
| Gain only | 0 (0) | 463 | 86 | 64 |
| Immediate | 345 (0) | 295 | 172 | 138 |
| Wait one query | 142 (142) | 394 | 120 | 92 |
| Learned immediate value | 322 (17) | 303 | 170 | 136 |
| Learned continuation value | 293 (37) | 313 | 168 | 132 |
| Immediate, matched to continuation usage | 293 (0) | 316 | 160 | 128 |

The primary comparison uses the same recordings, number of edits and
amplitude, with the delayed edits replayed at the original release query.
The continuation policy improves the conservative error reduction by
**+0.00160 pp [-0.00017, +0.00503]** (95% hierarchical bootstrap).
The comparison against the learned immediate-value head is also
inconclusive: +0.00146 pp [-0.00098, +0.00503]. Against constant velocity,
the continuation policy is +0.12961 pp [-0.00322, +0.24065].
The fixed wait leaves 203 credits unused; it should not be compared
with the full immediate policy as if both used the same budget.
No policy passes the original full-roster 2% recording-level rule.

## Timing and Recipient Capacity

For this diagnostic only, outcomes identify assignments that could pass
the unchanged joint rule. The first edits and all 345 second edits,
including amplitude, are preserved. Counts are out of 606 views.

| Available query | Top easy | All predicted easy | All legal |
| --- | ---: | ---: | ---: |
| Original release | 138 | 139 | 178 |
| Release or next two | 168 | 170 | 226 |
| Release or any later query | 198 | 199 | 249 |
| Any query, including before release | 198 | 199 | 255 |

Adding more tracks above the same easy threshold barely changes
capacity; considering all legal recipients changes it much more.
Widening the time window also changes capacity, but the two-query
learning experiment does not establish a timing benefit. A useful next
test is to change the recipient admission score, with the first edit,
number of edits and amplitude fixed. Hindsight counts cannot certify a
causal policy or replace undefined risk estimates.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_support_stop.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_support_stop.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_opendd_support_stop_menu.py
.venv-pytorch/bin/python scripts/freeze_m3w_dronecrowd_calibration_roles.py verify
```

`fresh_run`: 21 arm64 Torch fits and 8,400 updates; exact menu audit.
`cached_verified`: existing forecast bank and zero-update checkpoint
replay. `not_run`: independent DroneCrowd confirmation. Its frozen
allocation is 15 calibration groups and 30 independent confirmation
groups. No groups from that confirmation set were accessed by this
experiment. Deployment is unchanged; Stage5C and SMC remain off.
The study does not establish metric-scale or seconds-level SDD forecasts,
true 3D, foundation-scale modeling, or submission readiness.
