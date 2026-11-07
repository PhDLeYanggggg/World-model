# Supported Gain Within Fixed Forecast Families

Earlier cost-based retrieval sometimes dropped useful forecast
families. Here the original family slots, order, budget, forecasts,
direct actions and final risk-aware head remain fixed. Only the
agent representing each family changes. Two ridge outputs estimate
positive normalized supported gain and normalized positive risk
excess. The new within-family score subtracts the latter from the
former. The opening threshold is recalibrated using the unchanged
donor grid. The original 2% and reference support rules apply.

| First-two eligible queries before direct intervention | Old shortlist | Supported-gain shortlist |
|---|---:|---:|
| Useful action present, 18 original zero-direct views | 6 | 2 |
| Useful winner, 18 original zero-direct views | 0 | 0 |
| Useful action present, all 102 views | 9 | 10 |
| Useful winner, all 102 views | 0 | 6 |

The new shortlist retains two of the six zero-direct hits and adds
none. Although it keeps every family slot, it loses all three useful
damped-velocity-0.97 actions previously retained there. This separates
preserving a family from finding its useful action. Over the full
prefix cohort, more early winners do not imply an improvement to the
full policy.

| Policy | Calibrated switches | Complete passes / 102 | Signed easy >2% |
|---|---:|---:|---:|
| Direct only | 6,952 | 2 | 8 |
| Original shortlist selector | 6,955 | 2 | 8 |
| Supported-gain shortlist | 6,956 | 2 | 9 |

The calibrated effect is +0.00090 pp in paired completion-lower
utility, with 95% source interval [0, +0.00238]. At 6,998 switches on
the same queries, the effect is -0.00527 pp [-0.01359, -0.00006];
complete passes drop from 5 to 2 and signed easy violations rise
from 11 to 12. Bootstrap intervals use 3,000 resamples of 14 sources,
averaging three forecaster seeds within each source. The 102 views are
34 recordings repeated across the seeds.

## Why Six Early Winners Did Not Become Passes

A separate retrospective readout locates the earliest locally
useful winner in each of the six views: four at the first eligible
query and two at the second. In five views donor calibration returns
no eligible threshold; the sixth selects a different opening. In a
label-aware diagnostic only, adding each of those six useful
actions to its corresponding unchanged direct record still yields
**0/6** complete passes. This does not describe an inference policy.
It shows that for these recovered cases, better early ranking or
relaxing the threshold would not suffice to obtain a full pass
from that single action. The learned selector fails the registered
policy screen and is not promoted.

## Reproduction And Scope

Registration `3e641811` precedes the 42 source/seed ranker fits.
Each uses seven fitting sources, six calibration sources, and one
held source. Upstream forecasts and the final action head are
verified caches, not new training. With private parents available:

```sh
.venv-pytorch/bin/python scripts/run_m3w_family_supported_gain.py fit
.venv-pytorch/bin/python scripts/run_m3w_family_supported_gain.py score
.venv-pytorch/bin/python scripts/run_m3w_family_supported_gain.py replay
.venv-pytorch/bin/python scripts/audit_m3w_family_gain_openings.py
```

Scoring reproduces the previous control and the frozen readout
hash. The last command reads already-exposed labels solely for
diagnosis, without changing model choices. The task uses eight
observed and twelve predicted annotation steps in native image
coordinates. No independent outcomes were opened. No deployment,
physical safety, metric or seconds claim, Stage5C, or SMC.
