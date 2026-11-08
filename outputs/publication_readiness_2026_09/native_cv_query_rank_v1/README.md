# Choosing trajectory edits relative to CV

The earlier OLS4 ranker did not transfer to CV. I fixed the proposed
trajectories and changed how I train the selection score: one factor
is the reference (OLS4 or CV), the other is the loss (per-row utility
regression or regret-weighted ranking of two candidates from the
same query). The four arms see both reference displacements and use
the same initialization, training pairs and update count.

The ranking objective is a weighted logistic loss on score
differences, related to the established pairwise setup in
[Burges et al., ICML 2005](https://doi.org/10.1145/1102351.1102363).
Here the weight is the absolute difference in conservative
forecast utility, scaled once per query from the past. This
experiment tests the choice of objective, not a claim of ranking
algorithm novelty. [Registration](protocol.md)

## Same forecasts, same edit count

I fitted 108 new rankers (four treatments, three seeds, nine held
sources) with 500 updates each. The 108 inner donor models and 27
outer forecasters are verified cache. Each training proposal excludes
both its own source and the outer held source. There are 69,746
windows, with one edit per 1,176 queries in each seed. Scores use a
CV-error denominator, not relative FDE percentages. Intervals
resample nine sources 3,000 times after seed averaging.

| Ranker | Full-grid Lower Gain vs CV [95% CI] | Observed Gain |
|---|---:|---:|
| CV pair ranking | +0.0507 [-0.1573, +0.3053] | +0.2769 |
| CV row regression | -0.0314 [-0.3035, +0.2664] | +0.2769 |
| OLS4 pair ranking | -0.0886 [-0.3504, +0.1266] | +0.1803 |
| OLS4 row regression | -0.2128 [-0.4614, -0.0002] | +0.1273 |
| Prior OLS4 gain-and-harm ranker | -0.1707 [-0.4737, +0.0772] | +0.1748 |
| Minimum change from CV | -0.1843 [-0.3957, -0.0159] | +0.0440 |
| Observed-prefix rule | -1.0261 [-1.8136, -0.3223] | +0.3579 |
| Random | -0.5794 [-0.9304, -0.2924] | +0.2144 |
| Hindsight | +1.6843 [+0.8400, +2.7508] | +1.7259 |

The primary comparison does not establish positive gain. The CV
pair ranker improves the point estimate over CV regression, but
the direct paired full-grid lower contrast is -0.4084
[-0.5887, -0.2306]. Separate lower scores cannot be subtracted
to infer a paired advantage. The CV pair ranker's known worst-source
easy-harm ratio reaches 23.85%. That is a diagnostic, not an
independent safety calculation.

## Was the problem too much motion?

Keeping the CV ranker's nominations fixed, a quarter-strength edit
lowers the worst supported easy-harm ratio to 5.24% and produces
+0.0777 points [+0.0052, +0.1891] against no edit, positive in seven
of nine fitting sources. A past-only cap based on CV backtest
error gives +0.0236 [-0.0437, +0.1099] and up to 13.31% easy harm.
One source has no easy denominator in every seed. Neither approach
protects easy cases at the original 2% target.

At the same fixed quarter and edit count, the direct paired lower
contrast versus minimum displacement is +0.0047
[-0.0865, +0.1161], and versus CV regression it is -0.0854
[-0.1197, -0.0499]. The matched hindsight ceiling is +0.5258
[+0.2677, +0.8661]. These results distinguish a small same-source
gain over no edit from evidence that the learned ranker is better.
[Amplitude diagnostic](cap_protocol.md) |
[Matched quarter diagnostic](quarter_protocol.md)

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_cv_query_rank.py train
.venv-pytorch/bin/python scripts/run_m3w_cv_query_rank.py score
.venv-pytorch/bin/python scripts/run_m3w_cv_query_rank.py replay
.venv-pytorch/bin/python scripts/report_m3w_cv_query_rank_cap.py
.venv-pytorch/bin/python scripts/report_m3w_cv_query_rank_quarter.py
```

Use `train --resume` for existing checkpoints. Eleven targeted
tests passed. The [main score](analysis.json) and both diagnostics
([amplitude](cap_diagnostic.json), [matched quarter](quarter_matched_diagnostic.json))
reproduced exactly. Weights and individual trajectories stay private.
These are 8 observed and 12 forecast annotation steps in image
coordinates, not an independent test, metric or seconds claim.
The 15/30 external split and original 2% rule are unchanged. No new
deployment, Stage5C, or SMC.
