# Can Past Forecast Errors Improve The Final Action Choice?

**Result: the short-history backtest helps at matched coverage, but the
calibrated policy does not pass the registered improvement criterion.**
No model is promoted. This is exposed Moving A development evidence,
not independent external confirmation.

I fixed the eight-action shortlist, forecasts, original direct actions,
and 2% observed all/easy risk and signed easy requirements. I compared
the previous risk-aware selector with two new sets of fits: one adds
short CV errors measured inside the eight observed positions; the other
uses only older 12-step forecasts that have already resolved. The 84
new selector fits span 14 source-held cells and three forecaster seeds
per feature set. Each cell keeps seven fit sources, six calibration
sources and one held source. Candidate trajectories were not retrained.

| Selector | Calibrated switches | Complete passes / 102 views | Signed easy degradation >2% | Easy upper bound undefined |
|---|---:|---:|---:|---:|
| Direct only | 6,952 | 2 | 8 | 35 |
| Fixed shortlist, no feedback | 6,955 | 2 | 8 | 35 |
| Add 8-step-window backtest | 6,955 | 2 | 8 | 35 |
| Add resolved 12-step feedback | 6,959 | 2 | 9 | 34 |

Paired differences versus the unaugmented selector are in percentage
points of completion-lower utility, not relative FDE percentages.
Intervals use 3,000 resamples of 14 sources with the three seeds
averaged within source; 102 views are not 102 independent sources.

| Added input | Calibrated difference [95% interval] | Same queries and 6,998 switches [95% interval] |
|---|---:|---:|
| Short backtest | +0.00209 [0.00000, +0.00627] | +0.00525 [+0.00045, +0.01119] |
| Resolved feedback | +0.00027 [0.00000, +0.00081] | -0.00946 [-0.02841, +0.00016] |

In the first two eligible queries of the 18 views where direct never
intervened, **none** of the shortlisted actions had resolved own-agent
or recording feedback. Six views had a positive supported action in the
shortlist; all three selectors picked such an action in **0/18** views.
The backtest changes choice quality on some matched queries elsewhere,
but does not solve the opening-choice failure. The resolved-feedback
arm also adds a signed easy violation. This suggests testing richer
observed agent-neighbor motion for early choices, rather than waiting
for a long forecast to resolve or relaxing the threshold.

## Reproduction

From the repository root, with the prior private forecast and feedback
caches in place:

```sh
.venv-pytorch/bin/python scripts/run_m3w_shortlist_history.py fit
.venv-pytorch/bin/python scripts/run_m3w_shortlist_history.py score
.venv-pytorch/bin/python scripts/run_m3w_shortlist_history.py replay
```

The fit resumes at source/seed checkpoints. Registration
`532ec79f` precedes the real-data fits. The scoring code checks row
alignment, unchanged direct actions, identical diagnostic query counts,
and exact reproduction of the previous control counts and recall.
The full score replay reproduced the analysis hash. Forecast and
feedback parents are verified caches; the new selector fits and
readout are fresh. No protected outcomes were opened. Eight observed
and twelve future annotation steps are in native image coordinates,
not metric distance, seconds, or evidence of physical safety.
Stage5C and SMC remain off.
