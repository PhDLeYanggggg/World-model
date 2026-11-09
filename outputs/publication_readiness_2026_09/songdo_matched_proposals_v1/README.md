# Songdo: four matched two-edit proposals, with outcomes unopened

I ran four selection rules on the same unscored
[first-day vehicle bank](../songdo_first_day_v1/README.md). The heads
were trained on OpenDD, not on Songdo. I checked all 126 weight hashes
and source identities and averaged 21 heads per seed and architecture.
This step generated proposals, not a calibrated release policy.

For each flight and seed I used one past-defined eligible query and
held the forecast bank, exact edit count and displacement radius
fixed across all arms. The independent arm ranks actions separately;
the additive arm scores a pair without relative features; the
same-family arm requires the same forecast family for both edits;
the joint arm chooses freely among legal distinct-actor pairs.
The CV and OLS4 forecasts are unchanged.

| Seed | Eligible flights (of 200) | Exact unordered pair evaluations | Joint differs from independent | Joint differs from additive | Joint differs from same family |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 | 191 | 3,897,211 | 188 | 188 | 30 |
| 29 | 191 | 3,899,420 | 187 | 187 | 38 |
| 43 | 191 | 4,007,389 | 168 | 168 | 23 |

The same 191 flights are eligible for all three seeds, representing
all 17 candidate source components. There are 573 flight-seed
views, not 573 independent sources. The selected query is also the
same across seeds in 182 flights. The count differences only show
that the methods make different decisions; they say nothing about
accuracy or harm.

The largest checked query has 146 actors and 92,239 legal unordered
pairs. Recomputing the original joint models directly gives a maximum
logit difference of `5.68e-9` from the shared-embedding search and
reproduces all four saved trajectories exactly.

**Next:** freeze the approved source roles and release rule before
opening future coordinates. These equal-count proposals are a
utility comparison and do not certify the original 2% risk gate.
No future index or coordinates, thresholds, model promotion, new
training, Stage5C, SMC, or manuscript change.

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/build_m3w_songdo_matched_proposals.py
.venv-pytorch/bin/python scripts/verify_m3w_songdo_densest_proposal.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_songdo_matched_proposals.py tests/test_m3w_songdo_first_day.py tests/test_m3w_vehicle_pair_learning.py tests/test_m3w_vehicle_pair_support.py tests/test_m3w_songdo_candidate_bank.py
```

`fresh_run`: exact matched Torch inference and direct-search check.
Four local threads, no data workers, per-flight checkpoints, PID
heartbeat. Sixteen related tests passed. `cached_verified`: all
observed inputs, vehicle forecasts and OpenDD heads. `not_run`:
formal assignment, calibration and any outcome scoring. The NPZ
proposal packets stay local.

[Protocol](protocol.md) | [Proposal receipt](receipt.json) |
[Largest-query verification](densest_verification.json).
