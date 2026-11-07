# Does Within-History Forecast Validation Help At Twelve Steps?

I fitted velocity over two, four and six past positions, and varied
only how those forecasts were mixed. The control uses fixed prior
weights. The two adaptive versions use one- and two-step forecast
errors computed wholly inside the eight observations, either per agent
or averaged within the current query. No future labels are used to
select the weights, tune the temperature or generate any forecast.

| Contrast | Available-label gain (pp), 95% CI | Conservative lower gain (pp), 95% CI |
| --- | ---: | ---: |
| Matched recipients: agent vs fixed | +0.023 [+0.002, +0.048] | -0.120 [-0.391, +0.028] |
| Matched recipients: agent vs query mean | +0.011 [-0.004, +0.031] | +0.002 [-0.017, +0.025] |
| Matched recipients: agent vs original gain policy | -0.283 [-0.721, +0.186] | -1.081 [-2.612, +0.002] |
| Whole roster: agent vs OLS4 | +1.896 [+1.279, +2.488] | -1.291 [-4.482, +0.906] |
| Whole roster: agent vs last-step CV | -2.063 [-6.744, +2.616] | -12.944 [-25.311, -3.558] |

The last two rows are distinct whole-roster forecast comparisons, not
intervention-policy comparisons. Every percentage point uses observed
CV error as the denominator. Intervals use 3,000 paired source
resamples after averaging the three frozen producer seeds across
34 development recordings in 14 sources. No new stochastic fit was
run. A negative completion lower bound does not prove the unknown
full-label effect is negative.

The fixed, query-average and per-agent variants each pass **3/102**
of the original recording views versus **4/102** for gain ranking.
All keep exactly the same 9,675 genuine switches; their actual path
substitutions number 9,669, 9,669 and 9,672. All three leave 23 selected
easy ratios undefined. The worst signed easy degradation is 271.96%
for per-agent weighting versus 98.70% for gain ranking. Matched
conservative relative-path ADE gain for per-agent vs fixed is
-0.260 px [-0.731, -0.013]; the completion upper contrast is
+0.271 px [+0.016, +0.753], so the actual effect is not sign-identified.

**Decision:** short observed-prefix forecast errors offer a small
available-label signal, but they do not yield robust 12-step utility,
safer recordings or a demonstrated advantage to individualized over
query-shared weights. No new allocation study or deployment follows
from this arm. This diagnostic is in native image pixels and 8/12
annotation steps, not metric or seconds.

[Protocol](protocol.md) | [Registration](registration.json) |
[Aggregate](analysis.json) | [Past-only freeze](freeze_execution.json) |
[Exact replay](replay_execution.json).

```sh
.venv-pytorch/bin/python scripts/audit_m3w_prefix_velocity.py freeze
.venv-pytorch/bin/python scripts/audit_m3w_prefix_velocity.py score
.venv-pytorch/bin/python scripts/audit_m3w_prefix_velocity.py replay
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_prefix_velocity.py \
  tests/test_m3w_recording_blend_feasibility.py tests/test_m3w_joint_path_audit.py
```

10 focused tests passed. Freeze precedes label scoring; the local
arm64 score and full replay each took about 21 seconds after imports
and have identical output hashes. No new CREATE job, DroneCrowd role
access, independent result, Stage5C execution or SMC.
