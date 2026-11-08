# Does Short History Rank Future Forecast Directions?

I replayed the six frozen directions at observation prefixes of length
4, 5, 6 and 7. Each predictor receives the prefix, with only earlier
positions padded to meet its eight-step interface. It never receives
the observed suffix used to score that replay. At the actual forecast
origin all eight real observations are available; the chosen direction
comes from the unchanged 12-step bank. No post-origin positions, masks
or errors influence the choices. The linear rule averages replay gain
normalized by prefix peak radius. The agreement rule requires the same
winner and positive gain over matched hindcast under both linear and
constant-position padding. Otherwise it keeps the matched hindcast.

There are no new fits, tuned thresholds or calibration. All methods
share the same edited agents and realized peak radii; the unchanged
risk rule is applied after rollout. OpenDD uses the existing location-
excluded predictors (7 locations, 3 seeds, 30,075 edited row-seed
cases); TUMDOT uses the frozen ensemble (9 recordings in 1 campaign,
3 seeds, 4,368 edited row-seed cases). These are both exposed development
cohorts. TUMDOT tangent and hindcast controls reproduce exactly. The
bootstrap resamples seven OpenDD locations 3,000 times after averaging
seeds; no independent-domain interval is reported for TUMDOT.

| Half radius | OpenDD vs CV (pp) | OpenDD risk | TUMDOT vs CV (pp) | TUMDOT risk |
|---|---:|---:|---:|---:|
| Matched hindcast | +0.0056 | 552/606 | -0.0365 | 24/27 |
| Fixed tangent | +0.3057 | 522/606 | -0.0533 | 21/27 |
| Linear prefix | +0.2244 | 548/606 | -0.0396 | 24/27 |
| Padding agreement | +0.1355 | 553/606 | -0.0750 | 24/27 |

On OpenDD, linear replay gains +0.1736 pp [0.1549, 0.1938] directly
over hindcast but loses -0.0962 pp [-0.1276, -0.0767] directly to
fixed tangent. Agreement loses -0.2089 pp [-0.2402, -0.1711] to fixed
tangent. Thirty OpenDD views lack defined easy support. On TUMDOT,
both replay methods lose directly to matched hindcast, and smaller
edits pass 27/27 without beating CV. The balanced OpenDD pairwise
ranking concordance for linear replay is 49.8% [45.1%, 52.6%]; the
TUMDOT recording-average is 51.8%, descriptive only. Four replay
prefixes are available for every edited TUMDOT case and 98.8% on OpenDD.

## Does the recipient assignment help?

For each recording and seed, I permuted the actual multiset of six
path choices over the same edited recipients. The radius stays with
the recipient. Each permutation keeps the same count of each direction;
the expected additive utility over all permutations is computed exactly
by marginalization. This control tests assignment, not safety: no
permutation risk certificate is computed.

| Policy vs same-count shuffle, direct conservative utility (pp) | OpenDD [95% location bootstrap] | TUMDOT |
|---|---:|---:|
| Linear prefix | -0.0370 [-0.0584, -0.0202] | -0.0625 |
| Padding agreement | -0.0090 [-0.0384, +0.0432] | -0.1278 |

Direct comparison debits unobserved steps using the *difference* between
the compared predictions; it is not a subtraction of CV lower bounds.
The OpenDD linear rule also loses on observed steps alone (-0.0210 pp
[-0.0475, -0.0019]), so missing labels do not explain that result.
The gains over hindcast do not demonstrate that the tested past-only
replay assigns paths usefully. The four suffix lengths and synthetic
padding do not rule out causal predictors with more informative context.
Keep the deployment, risk rule and sealed confirmation groups unchanged.
Do not train a larger head on this unvalidated proxy as a main result.

```sh
.venv-pytorch/bin/python scripts/run_m3w_prefix_path_replay.py
.venv-pytorch/bin/python scripts/run_m3w_prefix_path_replay.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_prefix_path_assignment.py
```

[Replay registration](registration.json) | [Readout](analysis.json) |
[Matched permutation control](assignment.json).
