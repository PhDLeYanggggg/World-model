# Learning the Cost of Choosing a Motion Forecast

My previous backtest could often order candidates but chose a poor long-lead
forecast. I trained three small predictors to ask whether predicting the cost
of a choice is better than predicting the winning candidate. The candidates,
inputs, edited tracks, and peak edit on each track were identical. The result
is useful for choosing a training target, but it does not justify deployment.

| Contrast (right model improves left) | Observed gain, pp [95% source bootstrap] | Full-grid bound, pp |
|---|---:|---:|
| Pooled cost over winner classification | +0.478 [+0.133, +0.902] | [-2.308, +3.265] |
| Lead-block cost over pooled cost | -0.042 [-0.070, -0.012] | [-0.538, +0.454] |
| Pooled cost over unedited OLS4 | +2.388 [+1.162, +3.565] | [-14.044, +18.820] |

Gains are percentage points of the same CV error mass; the full-grid bounds
account for missing labels and are not confidence intervals. The bootstrap
95% interval on the primary **lower bound** is [-1.059, -0.186] pp. At lead
12 alone, pooled cost is +0.097 pp vs classification on observed labels,
with a -6.916 pp lower bound. Lead-block cost adds no observed gain at lead 12
(-0.034 pp vs pooled cost).

The winner classifier identified the known-label best action more often
(58.09% vs 57.44%). Pooled cost nevertheless reduced mean known regret from
1.464 to 1.372 pixels per labeled step on edited rows. The hindsight menu
oracle is +6.86% relative to OLS4 on those rows, not a deployable forecast.
The pooled model also reduced the observed positive harm mass at every lead
vs classification. It did not pass the original 2% rule: all three variants
have 0/102 passes. The chosen comparison forces edits on eligible rows to
isolate path choice, not to emulate a safe admission rule.

Why splitting cost did not help: the decision still sums all 12 steps. On
fully observed labels, the expected total is the sum of the expected lead
costs, so per-block prediction can help only through better estimation.
On the available labels it did not, and the missing-label bound got worse.
Learning costs rather than order is a reasonable component to carry forward;
simply adding more lead heads is not supported. The stronger follow-up is to
test whether the same cost head can reject poorly supported edits at a matched
query-level budget without losing the gains on known labels.

**Evidence scope.** `fresh_run`: nine native arm64 Torch fits, 9,000 optimizer
steps, 67,734 eligible fitting rows from nine groups, three seeds. On the
125,540 previously exposed development rows, each arm made exactly 125,488
edits per seed (376,464 across seeds), of which 122,665 per seed had any label
and 92,770 had a lead-12 label. Three seeds were averaged before the 3,000-draw
14-source bootstrap; 34 recordings are not 102 independent observations.
`cached_verified`: unchanged fit roster, OLS4/CV risk definitions, and exact
zero-update action/score replay. `not_run`: independent confirmation,
deployment, Stage5C or SMC. DroneCrowd still uses 15 calibration groups and
30 sealed independent confirmation groups. This is obs8/pred12 in annotation
order and local-image pixels, not seconds, metric, true 3D, or foundation.

```sh
.venv-pytorch/bin/python scripts/run_m3w_motion_choice_cost.py
.venv-pytorch/bin/python scripts/run_m3w_motion_choice_cost.py --replay
.venv-pytorch/bin/python scripts/build_m3w_motion_choice_cost_replay.py
```

The first command also resumes automatically at the last 100-step checkpoint.
The replay requires the original verified fit and A assets and does not train.
The anonymous ZIP reproduces the grouped tables from scalar sufficient
statistics without the assets; it is not a training release.
The native CPU run (four compute threads, zero workers) took 14.91 s at
2.64 GB maximum resident size. Replay took 9.60 s with zero updates; 29
focused tests passed. The anonymous ZIP was extracted and replayed outside
the repository with SHA-256
`d80dbb40c2f648b44ce4b6357b368de75cc01e514bf2924e9590a2630ed0e32c`.
[Protocol](protocol.md) | [Readout](readout.json) | [Freeze](freeze.json) |
[Supplement v121](../evidence_manuscript_v121/README.md)
