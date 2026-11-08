# Predicting Future Label Availability from Causal Geometry

The [budget experiment](../budget_risk_v1/README.md) found that increasing
the number of edits can improve observed utility without improving its
conservative lower bound. This experiment tests whether explicitly
predicting wholly missing future outcomes helps at the same edit budget.

On the same 14 Moving A development sources, 34 recordings, and three
forecaster seeds, I fit one cost model conditional on observed outcomes
and separate row-level missingness heads. Both heads use observed motion;
the geometry head also uses relative position, peer distance, crowd size
and outward motion. Its expected cost adds the legal action's known
disagreement radius weighted by predicted missingness. Real future masks
are used only for fitting labels and scoring. All primary arms match the
entangled policy's realized edits at each query, not just in aggregate.

| Rule | Edits | Wholly unlabeled edits | Joint passes | Risk passes | Undefined risk |
| --- | ---: | ---: | ---: | ---: | ---: |
| Entangled cost screen | 12,235 | 377 | 1 | 2 | 29 |
| Old gain rank | 12,235 | 733 | 1 | 1 | 32 |
| Conditional cost, no missingness term | 12,235 | 669 | 0 | 0 | 31 |
| Motion missingness | 12,235 | 983 | 2 | 4 | 28 |
| Geometry missingness | 12,235 | 287 | 1 | 2 | 29 |
| Geometry, free budget (diagnostic) | 8,579 | 130 | 2 | 5 | 29 |

Geometry improves the source-weighted Brier score from 0.03497 to
0.03348 (paired change -0.00149 [-0.00227, -0.00093]). Despite
90 fewer edits on unlabeled rows, it loses to the entangled screen
by -1.634 pp [-2.687, -0.659] in conservative paired utility. It
also loses to gain ranking (-2.749 pp [-4.416, -1.344]), motion
missingness (-2.752 pp [-4.185, -1.382]) and no missingness
(-2.832 pp [-4.502, -1.350]). Even the free-budget diagnostic
loses to the entangled screen (-1.372 pp [-2.476, -0.440]).
None gives a joint pass among the 36 views in which OLS4 is
certified better than CV. Intervals use 3,000 paired source
resamples after averaging seeds. The views are repeated, not 102
independent sources.

More accurate prediction of missingness, and fewer edits on wholly
unlabeled rows, are not sufficient for better trajectory decisions.
This comparison does not isolate the conditional cost model from
its ranking and screening rule. The next useful diagnostic is to
separate matched differences in observed gains, disagreement radii,
and forced budget filling before fitting another missingness head.
No policy was promoted. Original 2% and undefined-risk handling
remain in force; no new independent confirmation was run. DroneCrowd
uses 15 calibration groups and 30 sealed independent confirmation
groups. No metric/seconds claim, Stage5C or SMC.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_observability.py
.venv-pytorch/bin/python scripts/run_m3w_moving_observability.py --replay
.venv-pytorch/bin/python scripts/build_m3w_moving_observability_replay.py
```

`fresh_run`: 42 fits, 21,000 updates, 45.365 seconds, PID 84620,
four arm64 Torch threads, zero loader workers. Checkpoints and
heartbeat every 100 updates; rerun the same command to resume.
`cached_verified`: frozen parents and zero-update replay (24.151
seconds, PID 84745). `not_run`: new CREATE jobs or independent
confirmation. Run details and hashes are in [registration](registration.json),
[head freeze](head_freeze.json), and [analysis](analysis.json).
