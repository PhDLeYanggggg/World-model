# Independent Gain and Support Slots

I retained the sparse gain policy unchanged and gave each recording
one distinct optional support edit. The gain edit keeps amplitude
0.25. The support edit uses amplitude 0.10 at the first query with an
available recipient above the frozen easy threshold of 0.50. If both
slots are used at the same query, gain takes precedence and their
recipients must differ. Neither slot uses terminal information.

The main matched control spends the second slot in exactly the
same recording and query at amplitude 0.10, but ranks recipients
by predicted gain. The gain slot is byte-identical across these
policies. This is a separate two-slot development budget; it does
not replace the original fixed-budget evidence or the 2% rule.

## Results

| Policy | Edits | Undefined | Risk pass | Joint pass |
| --- | ---: | ---: | ---: | ---: |
| Gain alone | 598 | 463 | 86 | 64 |
| Support alone | 345 | 364 | 161 | 103 |
| Independent gain and support | 943 | 295 | 172 | 138 |
| Gain-ranked second at the same query | 943 | 341 | 142 | 113 |
| Support amplitude 0.25, same recipients | 943 | 295 | 153 | 123 |
| Gain rank among predicted easy, same query | 943 | 293 | 173 | 139 |
| Outcome-aware recipients, same queries | 943 | 277 | 208 | 178 |

Joint passes require the original recording risk checks and positive
lower gain over both CV and OLS4. At matched counts and queries, the
support-ranked policy selects 242 truly easy second recipients versus
180 for the gain-ranked control. It gains 49 joint passes and loses
24 relative to that control. Against gain alone it gains 88 passes
and loses 14. The simpler gain-ranked arm preserves more utility.

| Direct lower-gain contrast | pp [95% site-bootstrap interval] |
| --- | ---: |
| Gain alone vs CV | +0.143 [+0.031, +0.241] |
| Two roles vs gain alone | -0.014 [-0.040, +0.00002] |
| Two roles vs same-query gain rank | -0.034 [-0.054, -0.015] |
| Two roles vs CV | +0.128 [-0.004, +0.239] |
| Same-query gain rank vs CV | +0.161 [+0.032, +0.275] |
| Support amplitude 0.25 vs 0.10 | -0.024 [-0.063, -0.002] |
| Rank gain among predicted easy vs easy rank | +0.00002 [-0.00010, +0.00017] |

The gain-within-easy follow-up uses the same eligibility and all
345 second-slot queries. It changes only 11 recipients. Raising
support amplitude also does not help. Relative-position lower
gain for two roles versus gain alone is +0.014 pp [-0.001, +0.038];
it is a forecast error proxy, not physical safety. Two-role mean
easy net degradation is 0.65%, but 295 undefined views and 172
risk passes make clear that the original full-roster rule fails.
The worst-site lower gain for the same-query gain arm is -0.127 pp
versus CV. The seven-site bootstrap averages three seeds before
each of 3,000 draws; the 606 views repeat 202 exposed recordings.

## What Timing Could Still Change

Keeping the first action, the second-slot recording set, its count
and amplitude fixed, I enumerate the recipient of the second slot
with outcomes known. At the original query, 178 views can pass.
When the second query can change within the same recording, the
exact bound is 255. Each added witness passes the original scorer.
The 261 views without a second slot are unchanged, and 24 of those
already pass. The difference of 77 isolates a limitation of the
current second-slot query, not a causal gain or a change of budget.

The second slot improves support, but more suitable recipients at
the same query are not enough. I would next test a causal policy
for when to use the support slot, retaining the gain-ranked arm
and matching realized coverage. The current result does not
advance to independent confirmation or deployment.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_two_reserves.py
.venv-pytorch/bin/python scripts/audit_m3w_opendd_two_reserve_ranking.py
.venv-pytorch/bin/python scripts/audit_m3w_opendd_two_reserve_timing.py
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py --include-two-reserves
```

`fresh_run`: matched two-slot inference, within-query reranking
and exact timing capacity. `cached_verified`: original forecasts,
release and gain heads; all three readouts reproduce exactly.
Native arm64 Torch with four threads and zero loader workers.
No training, CREATE job, Stage5C, SMC or deployment. DroneCrowd
15/30 roles were verified without opening confirmation outcomes.
Dataset-local coordinates and 8/12 annotation steps.

[Main results](analysis.json) | [Reranking](ranking_analysis.json) |
[Timing capacity](timing_capacity.json)
