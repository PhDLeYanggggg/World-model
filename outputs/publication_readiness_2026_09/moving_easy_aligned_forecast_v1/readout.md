# Easy-Aligned Forecast Training Reduces One Harm Measure, Not The Full Loss

I continued the three original fold-2 PCT predictors from their verified
4,000-update checkpoints. Both arms receive the same 300 new updates per seed,
architecture, optimizer initialization and batch stream. The treatment changes
only the objective: add twice the observed easy ADE excess over causal CV.
Fit data are the original 69,746 eligible rows from 24 recordings, nine
nonempty source groups and one declared zero-support group. The 12.4916 px
positive-easy training cutoff is derived on these fitting labels only. The
previously frozen held easy/hard cutoffs and original 2% rule do not change.

| Treatment minus matched ADE continuation | Point (pp) | 95% exploratory source interval |
|---|---:|---:|
| Known-label utility | +0.747 | [-0.666, +2.205] |
| Completion-lower utility | -7.675 | [-16.963, -2.093] |
| Known easy positive-harm change | -6.976 | [-15.513, -0.610] |
| Known hard utility | +0.092 | [-0.601, +0.809] |

All 125,540 A rows, 34 recordings and 14 sources are retained for each actual
seed; the source bootstrap averages seeds before resampling. These are raw
forecast comparisons, not a newly selected or calibrated intervention policy.
Easy positive harm is normalized by the full known easy CV reference, not by
selected reference under the operational 2% gate. It falls from 220.09% to
213.12% in this source-macro diagnostic, still very large. The seed-specific
harm changes are -11.02, -10.27 and +0.36 pp. The hard interval does not prove
improvement or noninferiority. Against CV the raw known utility remains
negative in both arms (-22.67% and -21.93%).

**Do not promote the candidate.** The separation of the two predictions on
wholly unobserved rows costs 8.42 pp between the known contrast and its lower
bound, especially in one high-disagreement source. That is not observed harm.
Adding either warm-continued path to the broader frozen action bank leaves the
label-aware ceiling at 101/102 views. The same recording and seed lack an easy
witness; no view is removed or certified by equality. A useful follow-up would
target unobserved-row disagreement using a past-only trajectory agreement loss
on the fitting set, rather than strengthen the easy hinge or tune held cuts.
Such a loss would use training availability only as supervision, not as input.

## Reproduction And Cost

```sh
.venv-pytorch/bin/python scripts/run_m3w_easy_aligned_forecast.py prefix
.venv-pytorch/bin/python scripts/run_m3w_easy_aligned_forecast.py train
.venv-pytorch/bin/python scripts/run_m3w_easy_aligned_forecast.py infer
.venv-pytorch/bin/python scripts/run_m3w_easy_aligned_forecast.py score
.venv-pytorch/bin/python scripts/run_m3w_easy_aligned_forecast.py replay
```

The prefix phase writes the initial 20 steps of the first arm only; the full
train invocation resumes them. Those steps count within the 1,800 new updates,
not in addition. All arms have the same final draw counts and RNG state within
each seed. Source-ID validation repair is registered before any update.
`prefix`/`train`/`infer`/`score`/`replay` all exit 0 (PIDs 15266, 15297,
15387, 15573, 15619). Full continuation after the prefix takes 38.70 s;
all 753,240 fresh A path predictions take 122.51 s. Every complete prediction
bank freezes before A labels enter this experiment. Scoring replay is exact
on the frozen outputs, not a second fit or second model forward. Native arm64
Torch CPU4, one inter-op thread, no loader workers. No new external labels,
independent confirmation, deployment, metric/seconds, Stage5C or SMC.

[Registered experiment](protocol.md) | [All scalar evidence](analysis.json)
