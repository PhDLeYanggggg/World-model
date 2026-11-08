# Does It Help to Wait for the Gain Edit?

The [fixed-support audit](../opendd_protected_gain_v1/README.md) found
headroom when the gain slot could move later, but it used outcomes to
choose the best time. Here a head decides once, at the original gain
trigger. Waiting means accepting the first eligible offer among the
next eight query indices, not searching future results. Support must
have been observed. An offer uses the original gain ranking among
current non-support rows that can be scaled down to exactly the
original gain radius. An expiry leaves only the fixed support path.

The head predicts realized conservative wait-minus-now utility,
loss of a defined reference/risk pass/joint pass, and expiry. The
donor labels compare the outcomes of two fixed policies rather than
the best hindsight action. The guard requires positive estimated
utility and negative deterioration and expiry logits. The
anchor-only guard is retrained without support/prefix context.
Fit on six development locations, score on the seventh, for three
seeds. Upstream donor trajectories and nomination heads exclude both
the donor and evaluation locations.

| Policy | Edits | Joint pass | Risk pass | Undefined | Waits | Expiries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original immediate | 943 | 138 | 172 | 295 | 0 | 0 |
| Always wait | 558 | 109 | 146 | 349 | 598 | 385 |
| Utility head | 797 | 107 | 143 | 321 | 248 | 146 |
| Guarded head | 841 | 111 | 146 | 317 | 189 | 102 |
| Immediate, guard-matched | 841 | 124 | 157 | 318 | 0 | N/A |
| Anchor-only guard | 859 | 114 | 155 | 311 | 170 | 84 |
| Immediate/first-offer oracle | 943 | 174 | 203 | 285 | 98 | 0 |

The 606 recording-seed views come from seven exposed development
locations. The matched immediate comparator retrospectively retains
the same gain and support counts in each recording, not just the
aggregate. It is a diagnostic control, not an online alternative.
The guarded head gives -0.03396 pp [-0.05847, -0.01133] versus the
original, and -0.01851 pp [-0.04515, -0.00201] versus the matched
control. Full context versus retrained anchor-only is -0.02956 pp
[-0.05589, -0.00501]. Intervals are 3,000 paired location resamples
after averaging seeds. The current context does not improve timing.

## What Expires and What Could Improve

Of 385 expiries, 257 never see the original support edit, 62 see it
after the window, and 66 see support but lack a radius-matched
neural offer. Other causal directions would supply a candidate in
58 of the 66 radius cases, not necessarily a passing candidate.

| Outcome-aware menu, retaining the original as an option | Joint pass |
| --- | ---: |
| Wait only if support is already observed at release; first neural offer | 159 |
| Immediate or first eligible neural offer | 174 |
| Any recipient at the first eligible neural query | 176 |
| Any neural query and recipient in the window | 197 |
| First query with any matched direction; any recipient or direction there | 194 |
| Any query/recipient/direction in the same window | 238 |

The direction-expanded menu adds matched OLS4 and observed-prefix
directions, never larger edits. Every new feasible witness is checked
with the original scorer and must preserve an old defined reference.
The first-offer oracle's 36 added joint passes include 26 from one
location. These outcome-aware counts diagnose action availability
and concentration; they are not causal policy results.

The largest limitation is missing or late support, not the ranking
of recipients at the first neural query (174 to 176). A useful next
test would learn where early, reliable support is available on
development data and compare it with the original at matched edit
counts and radii. More waiting without a new support mechanism
cannot resolve the 257 never-supported releases in this design.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_gain_wait.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_gain_wait.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_opendd_gain_wait.py
```

`fresh_run`: 42 native arm64 Torch fits, 21,000 updates, four
threads, zero loader workers, 22.6 seconds (PID 76874). Checkpoint
and heartbeat every 100 updates; the same command resumes.
`cached_verified`: frozen roles, forecast and head receipts;
zero-update replay. `not_run`: new CREATE training and independent
confirmation. The original 2% per-recording rule is unchanged;
undefined risk does not pass. DroneCrowd stays at 15 calibration
groups and 30 sealed confirmation groups. No deployment change,
submission readiness, Stage5C, or SMC.
