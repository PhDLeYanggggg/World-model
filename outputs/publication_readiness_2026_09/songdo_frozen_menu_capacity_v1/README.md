# How much gain is left in the frozen Songdo proposals?

The frozen joint rule released zero forecasts. I checked whether the
problem was that the four saved proposals had nothing useful to offer or
that the release rule missed the available choices. This is a post-hoc
diagnostic on the same already scored confirmation views. No outcomes
were reopened and nothing was fitted on them.

For each flight and seed, allow the unchanged CV fallback or one of the
specified frozen proposals. Each proposal uses the same query, two edits
and the same peak radius. Keep only proposals that pass the three original
2% risk checks and have strictly positive conservative gains against CV
and OLS4. Take the highest conservative CV gain among these proposals, or
zero if none qualifies. Zero means keep CV; it is not a risk pass.

| Available proposals | Qualifying views / 480 | Flights with any / all three seeds qualifying | Source-averaged CV gain ceiling (pp) |
| --- | ---: | ---: | ---: |
| Independent only | 169 | 95 / 18 | +0.00021013 |
| Additive only | 162 | 92 / 15 | +0.00019440 |
| Shared family only | 56 | 43 / 2 | +0.00015937 |
| Joint only | 53 | 45 / 0 | +0.00011320 |
| All except joint | 193 | 102 / 23 | +0.00025230 |
| All four | 197 | 105 / 23 | +0.00025702 |

The full menu has qualifying views in all 14 components. Its 95% component
bootstrap interval is [+0.00008786, +0.00055582] pp from the original 3,000
draws. Adding joint to the three other arms enables four new qualifying
views and raises the ceiling in five views in total. The increase is
+0.00000472 pp, an arithmetic increase between two CV-relative conservative
scores, not a paired lower bound on the gain of the joint policy. A larger
menu cannot hurt the outcome-aware optimum. A bootstrap interval for that
increment is therefore not evidence of a learned method improving.

The joint rule leaves all 53 of its qualifying proposals unreleased. The
independent rule releases 15 views and captures four of its 169 qualifying
proposals. There is a decision gap, but perfect hindsight selection would
still make a very small difference on the whole flight under this budget.
These are limits of the frozen finite menu and the original conservative
score, not upper bounds for every possible trajectory model or future
completion. They do not change the original confirmation or justify release.

```sh
python3 -I -B -S scripts/audit_m3w_songdo_frozen_menu_capacity.py
```

The script reads the hash-bound local v3 supplementary archive, reconstructs
the pre-outcome thresholds and original release counts, then calculates the
menu bounds. It does not need the datasets or model weights. The full
[aggregate receipt](receipt.json) keeps per-menu intervals, repeated-seed
support and actual release counts. No private flight identifiers are exported.
