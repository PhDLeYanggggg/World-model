# A Prospective One-Edit Reserve

I made one optional extra edit available from the start of each
OpenDD recording. The reserve is distinct from the original fixed
budget. It never removes an incumbent edit or assigns two actions
to the same row. When used, it applies 0.25 of the neural forecast's
displacement from CV. It sees only current and earlier queries;
there is no end-of-recording release. The original 2% risk rule
and both lower-gain comparisons are unchanged.

The fixed easy rule releases at the first query with an eligible
candidate having easy probability at least 0.50. The gain rule
instead requires positive predicted lower CV gain. The joint head
also requires a predicted probability of at least 0.25 for a
local event: easy reference, positive lower CV gain and no
all/easy risk excess for the individual action. This event is not
a recording-level certificate. Fitting forecasts exclude both the
donor and outer site. No thresholds were selected on held outcomes.

## Full Policies And Matched Recipients

| Rule | Extra edits | Undefined | Risk pass | Joint pass |
| --- | ---: | ---: | ---: | ---: |
| No reserve | 0 | 401 | 26 | 16 |
| Easy probability | 345 | 246 | 97 | 68 |
| Predicted gain | 598 | 288 | 77 | 56 |
| Joint event | 79 | 384 | 31 | 19 |
| Easy, same query as joint | 79 | 382 | 30 | 18 |
| Gain, same query as joint | 79 | 391 | 30 | 19 |

Joint pass means original risk and support pass with positive
lower gains over CV and OLS4. Full rules differ in realized
coverage. Only the same-query controls match extra counts,
recordings, query times, and amplitude; recipient identity may
differ. The gain rule improves conservative utility over the
incumbent by +0.141 pp [+0.036, +0.239], but does not beat CV:
-0.048 pp [-0.427, +0.287]. The easy rule increases support
while decreasing utility versus the incumbent by -0.038 pp
[-0.103, -0.002]. The joint event loses -0.006 pp
[-0.013, -0.0004] against the incumbent and does not reliably
beat easy ranking at the same query.

Direct CV effects are also computed for all arms. For the
reconstructed values, the paired lower bound is additive here
because incumbent and reserve edits occupy disjoint rows and share
the same CV normalization. The direct joint-versus-CV calculation
verifies that identity in every site-seed cell.

## Follow-Up: Use Both Existing Signals

I tested a separate factorized rule after the first readout, with
no new training and no new thresholds. It uses the first query
where a candidate has easy probability at least 0.50 and positive
predicted lower gain, then ranks the eligible candidates by gain.

| Rule | Extra edits | Undefined | Risk pass | Joint pass |
| --- | ---: | ---: | ---: | ---: |
| Factorized | 207 | 319 | 66 | 47 |
| Easy, same query | 207 | 321 | 62 | 43 |
| Gain, same query | 207 | 337 | 58 | 43 |

Its paired lower gain versus easy at the exact same queries is
+0.00024 pp [0.00000, +0.00069]. Against gain at those same queries
it is -0.0227 pp [-0.0442, -0.0040]; against no reserve it is
-0.0012 pp [-0.0061, +0.0054]. Against CV the effect is -0.191 pp
[-0.619, +0.161]. Intervals resample seven sites after averaging
three seed views, with 3,000 draws. The 606 views repeat 202
exposed recordings; they are not independent confirmation.

## What the Slot Recovered

| Cohort assigned after the run | Views | Quarter-amplitude hindsight | Easy rule | Gain rule | Joint rule | Factorized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Incumbent has zero edits | 244 | 138 | 32 | 28 | 3 | 20 |
| Edits but no reachable easy reference | 68 | 26 | 8 | 3 | 1 | 7 |

These cohorts are never input to any policy. The easy-only and
gain-only mechanisms help different endpoints; requiring both
does not recover their strengths at once. I retain the incumbent
decision. A useful next comparison holds the selected query and
recipient fixed and tests the candidate change itself against a
bounded analytic alternative, rather than searching reserve thresholds
on these exposed sites.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_reserve.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_reserve.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_opendd_factorized_reserve.py
.venv-pytorch/bin/python scripts/audit_m3w_opendd_reserve_absolute.py
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py --include-reserve
```

`fresh_run`: 21 arm64 Torch fits, 10,500 optimizer updates,
11.3 seconds; four threads, zero loading workers, checkpoint and
heartbeat every 100 steps. The original command resumes.
`cached_verified`: the prior candidate bank, incumbent and release
heads; 6.8-second zero-update replay matches. The initial runner's
long legacy import chain hit Python's recursion limit before
registration or fitting; explicit dependency ordering fixed it.
Thirty-two focused tests passed. No new CREATE job.

Native 8/12 annotation steps and dataset-local coordinates. No
deployment, Stage5C or SMC. DroneCrowd's 15/30 roles remain frozen;
confirmation outcomes were not opened.

[Registration](registration.json) | [Results](analysis.json) |
[Factorized follow-up](factor_analysis.json) | [Direct CV effects](absolute.json)
