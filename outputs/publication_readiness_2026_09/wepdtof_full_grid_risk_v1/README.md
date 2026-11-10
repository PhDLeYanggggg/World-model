# WEPDTOF: passing risk does not imply there is a gain

I applied the same full-grid risk calculation used on Songdo to
the 15 previously scored WEPDTOF clips from 13 sources. The four
matched arms keep the same forecasts, label matches, counts, motion
references and 2% release rule. Each arm edits 36 of 45 clip-seed views
and makes 366 edits. The other nine are not certified passes.

| Matched arm | Original risk passes | Passes risk for all twelve-step completions | Of those, utility excluded even with favorable missing positions | Utility excluded among all 36 edited views |
| --- | ---: | ---: | ---: | ---: |
| Independent | 4 | 4 | 3 | 21 |
| Common family | 3 | 3 | 2 | 21 |
| No pair penalty | 4 | 4 | 3 | 20 |
| Joint | 4 | 4 | 3 | 20 |

The joint rule retains the same four risk passes. For three, the
optimistic utility endpoint is still negative against at least one
reference. This rules out beating both CV and OLS4 under any missing
position completion. The remaining risk pass is unresolved, not a
success. Conversely, one edited joint view is guaranteed to beat
both references but does not pass the full-grid risk check. None of
the four arms has a guaranteed joint risk-and-utility pass. The
original zero dual passes and the frozen release decision are unchanged.

Guaranteed easy support is 24 joint versus 25 independent views.
The paired source-weighted difference is -3.03 percentage points
(95% bootstrap interval [-12.12, +6.06]; 11 sources with edits,
3,000 resamples). This does not reproduce the large descriptive
easy-support gap in Songdo. For WEPDTOF, the more direct limitation
is utility against the two motion references at this budget.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_wepdtof_full_grid.py
```

`fresh_run`: full-grid risk and utility sensitivity from already scored
numeric labels. `cached_verified`: the pre-outcome forecasts, original
risk and utility values, label matching and cutoff. `not_run`: new
confirmation, fitting or deployment. WEPDTOF remains in image pixels;
these are not physical risk or time-calibrated estimates.
[Protocol](protocol.md) | [Readout](readout.json) |
[Dataset paper](https://openaccess.thecvf.com/content/WACV2022/html/Tezcan_WEPDTOF_A_Dataset_and_Benchmark_Algorithms_for_In-the-Wild_People_Detection_WACV_2022_paper.html).
