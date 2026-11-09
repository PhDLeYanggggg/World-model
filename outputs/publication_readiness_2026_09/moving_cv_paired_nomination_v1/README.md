# Can a Conservative Preference Loss Improve Trajectory Selection?

I trained two rankers to test whether learning the preference between
candidate paths helps beyond predicting each path's value separately.
Both predict the original two CV-anchored gain terms and see the same
trajectory geometry relative to CV and OLS4. The new term compares two
paths for the same agent. It adds a ranking gradient only when the
known-step advantage exceeds the maximum possible loss on missing steps.
The original risk model, candidate library, and number of edits in every
query are held fixed in the diagnostic comparison.

| Held fitting-source strategy | Edits | Original risk passes /72 | Undefined easy /72 | Worst known signed easy change | Conservative utility vs. CV (pp) |
|---|---:|---:|---:|---:|---:|
| Frozen full-menu ranker | 2,317 | 9 | 39 | +32.32% | +0.0074 |
| Matched pointwise ranker | 2,317 | 14 | 36 | +32.11% | +0.0164 |
| Matched preference ranker | 2,317 | 14 | 38 | +0.84% | +0.0391 |
| Either calibration-selected policy | 0 | 0 | 72 | Undefined edited support | 0 (identity) |

**Less known easy tail harm, but no verified utility gain or admission.**
Preference versus pointwise gives a direct paired lower utility of
-0.0368 pp, with a 95% source interval of [-0.1060, +0.0072].
The separate bounds against CV are not additive and cannot be subtracted
to infer a paired effect. The preference arm's bound against OLS4 is
-55.6732 pp [-97.2280, -23.4307], in units of observed CV error mass.
On 445 different paths chosen for the same recipient, preference wins
168, pointwise wins 221, and 56 are not resolved by the bound. Those
agents are not independent observations.

The preference arm has no signed known easy changes above 2%, versus
two for pointwise. But 38 views have no easy reference, and the all-view
conservative harm bound exceeds 2% in 18 views. These counts can
overlap. Each ranker qualifies **0/9** calibration cells; at most three
recordings in a cell pass. Neither method replaces the CV fallback.
The remaining bottleneck is the recording-level availability of useful,
reference-supported, low-harm edits, not merely the path ranking
objective. The next comparison should target query-level allocation of
the existing edit menu under that support constraint.

Decision-focused learning motivates the comparison, but the loss is not
[SPO+](https://arxiv.org/abs/1710.08005), and this study does not inherit
that paper's consistency result. Unlike the class decisions and expert
deferral studied by [Mozannar and Sontag](https://proceedings.mlr.press/v119/mozannar20b.html),
this is a geometric choice over a per-agent path library with a fixed
motion fallback and a separate recording-level admission test.

## Reproduction

`fresh_run`: 18 native arm64 Torch fits, 18,000 updates, 105.52 s;
three seeds, 24 held fitting recordings, and nine source groups.
Checkpoint every 100 updates, four CPU threads, zero loader workers.
Intervals resample sources 3,000 times after seed averaging.
`cached_verified`: risk heads, neural donor forecasts, prior matched
choices, and exact zero-update replay. Nineteen focused tests pass;
the full legacy suite was not rerun.

```sh
.venv-pytorch/bin/python scripts/run_m3w_cv_paired_nomination.py
.venv-pytorch/bin/python scripts/run_m3w_cv_paired_nomination.py --replay
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_cv_{paired_nomination,screen_order,admission,joint_selector}.py tests/test_m3w_{grouped_recording_readout,selection_forecast_readout}.py
```

`not_run`: new A readout, new DroneCrowd calibration, independent
confirmation, deployment, Stage5C, SMC. The original 2% rule and
15/30 split remain in effect; all 30 confirmation groups remain sealed.
Native 8/12 annotation steps and local image pixels are not verified
seconds or physical units. This is not 3D or a foundation model.

[Protocol](protocol.md) | [Metrics](readout.json) |
[Manuscript v127](../evidence_manuscript_v127/README.md)
