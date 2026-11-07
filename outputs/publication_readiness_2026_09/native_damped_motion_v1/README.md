# Do Damped Speed And Heading Changes Add Useful Forecast Shapes?

This experiment extends the native bank with distinct motion shapes,
rather than interpolating the existing paths. The starting velocity
comes from OLS4. The new trajectories extrapolate a clipped median
backward speed change and, in the turning arm, a persistent signed
heading change. The increments decay with fixed half-lives of three
or eight annotation steps. At the same half-life the straight and
turning trajectories have the same speed schedule. The constants
and trigger rules were frozen in the [protocol](protocol.md) and
[registration](registration.json) before scoring.

There are two different questions. The first is whether the new
paths add to the label-aware capacity of the existing bank under
the original CV/OLS4 comparisons, 2% recording checks, and exact
query counts. The second is whether a simple history-only rule
benefits when substituting those paths for people whom the frozen
gain policy already chose to switch. The latter keeps the recipient
mask, hence every query count, exactly unchanged. It uses the same
frozen forecaster paths and adds no data-driven fitting.

## Results

| Nested bank | Verified both references | Negative maximin bound | Risk/count infeasible | Unresolved |
| --- | ---: | ---: | ---: | ---: |
| Native | 51 | 45 | 6 | 0 |
| Add speed trends | 51 | 45 | 6 | 0 |
| Add matched-speed turning paths | 51 | 45 | 6 | 0 |

There is no newly positive recording or source at the exact 9,675
switches. Both paired source-bootstrap verified-rate differences
are 0 pp [0, 0] on this finite development roster. The expansion
did not re-optimize already positive views, so this says nothing
about how much their utility could improve.

| Past-only path contrast, same recipients | Available-label change (pp) | Conservative lower change (pp), 95% CI |
| --- | ---: | ---: |
| Speed trend vs. original gain ranking | -0.567 | -1.920 [-4.772, -0.230] |
| Speed and turn vs. speed trend | -0.502 | -0.860 [-1.728, -0.346] |
| Speed and turn vs. original gain ranking | -1.070 | -2.778 [-5.634, -0.895] |

All effects use observed CV mass as the denominator. Intervals draw
14 source groups 3,000 times, averaging the three forecaster seeds
before resampling. The speed rule changes 3,833 paths; the turn
trigger changes another 3,830 relative to the speed rule (the sets
can overlap). Both tested rules pass 0/102 recording views and
exceed the all-risk bound in 102/102, versus 4/102 passes and
88/102 all-risk exceedances for the original matched gain control.

The registered turn trigger does not necessarily fire on the same
people as the speed trigger. A separate post-hoc
[heading-only contrast](matched_heading_analysis.json) sets both
arms to exactly the same speed at every forecast step and changes
only heading on the same 3,830 replacement rows. It still gives
-0.416 pp [-0.612, -0.247] on available labels and -0.859 pp
[-1.910, -0.257] conservatively. Its two arms also pass 0/102.
This tests extrapolating observed heading under these fixed triggers,
not the value of all curved predictions or all learned motion models.

[Primary analysis](analysis.json) | [Score receipt](score_execution.json) |
[Replay receipt](replay_execution.json). Twelve focused tests passed.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/audit_m3w_damped_motion_capacity.py score
.venv-pytorch/bin/python scripts/audit_m3w_damped_motion_capacity.py replay
.venv-pytorch/bin/python scripts/report_m3w_matched_turn_heading.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_damped_motion_options.py \
  tests/test_m3w_dual_reference_capacity.py tests/test_m3w_ols_blend_capacity.py \
  tests/test_m3w_missing_debit_capacity.py
```

The audit uses 34 exposed Moving A recordings from 14 source groups
and three frozen forecaster seeds. The deterministic new path generator
is not trained three times. Scores use eight observed and twelve
forecast annotation steps and native image pixels. A recording
repeated across seeds is not an independent scene. Positive capacity
witnesses and actual past-only predictions are rescored in replay;
replay does not rerun solver proofs. No external access, new training,
deployment, Stage5C or SMC.
