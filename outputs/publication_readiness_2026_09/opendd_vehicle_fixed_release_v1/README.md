# OpenDD: learn a better score without changing either forecast

The [previous audit](../vehicle_recording_replay_v1/README.md) found that
the high scores of the relative head did not identify its useful
two-edit proposals. I keep the candidate and additive reference paths
fixed and change only the release score. This isolates the release
decision from changes in recipient, path family or displacement.

The direct head learns the binary full condition with two auxiliary
labels: the original risk checks and the risk-plus-dual-reference
condition. The nested head estimates the same event as the product of
risk pass, dual-reference pass given risk, and full pass given both.
Both refits use the same 315 donor batches, architecture, initial
weights, minibatches and optimizer updates. Training uses conservative
gain signs as a surrogate; scoring uses the exact original tolerances.

## Ranking on identical candidates

Each arm ranks the same frozen proposal for every recording. The table
reports the proportion of qualifying proposals among the five highest
scores minus the exact expected proportion under uniform selection
from the same direction.

| Top five minus same-direction uniform, pp | Archived relative score | Direct | Nested |
| --- | ---: | ---: | ---: |
| Original risk pass | -4.23 [-7.35, -0.96] | +8.63 [+4.86, +12.82] | -1.05 [-5.81, +3.97] |
| Risk and gain over CV/OLS4 | -0.03 [-2.96, +2.71] | +4.10 [+0.07, +7.62] | +0.77 [-2.77, +4.41] |
| Full condition including additive contrast | -0.25 [-2.01, +1.79] | +1.81 [+0.17, +3.50] | -0.73 [-2.49, +0.65] |

On the direct paired contrast, the direct head adds **12.86 pp**
[7.14, 19.21] of risk passes and **2.06 pp** [0.16, 4.13] of full
passes within the top five versus the archived score. The risk contrast
is positive in all seven locations. Nested minus direct loses 9.68 pp
[-17.62, -1.74] on risk and 2.54 pp [-3.97, -1.27] on the full condition.
Intervals use 3,000 resamples of seven location means, after averaging
three seeds and six calibration choices. The 8,874 repeated views
represent 493 distinct recordings.

## Frozen release

| Held outcomes | Archived | Direct | Nested |
| --- | ---: | ---: | ---: |
| Directions with a nonempty calibration prefix | 9 | 7 | 6 |
| Directions releasing on held data | 4 | 2 | 0 |
| Admitted replacements | 14 | 11 | 0 |
| Undefined easy reference | 7 | 2 | 0 |
| Pass original risk | 6 | 5 | 0 |
| Pass risk and gain over CV/OLS4 | 5 | 5 | 0 |
| Meet the full condition | 1 | 1 | 0 |
| Active directions with every release qualifying | 0 / 4 | 0 / 2 | Not estimable |

The direct head improves ranking, but its original-rule calibration
still transfers poorly: four of its six risk failures have an
estimable easy denominator. Missing support is not the only remaining
failure. The nested head abstains on held data; that is not a risk pass.
These release counts do not compare equal coverage. I keep CV fallback
and do not promote either model or modify the main paper. No Songdo or
DroneCrowd confirmation outcomes were opened.

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_vehicle_fixed_release.py freeze
.venv-pytorch/bin/python scripts/run_m3w_vehicle_fixed_release.py score
.venv-pytorch/bin/python scripts/audit_m3w_vehicle_fixed_release_contrasts.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_vehicle_release_score.py tests/test_m3w_vehicle_recording_replay.py tests/test_m3w_vehicle_anchor_rank.py
```

`fresh_run`: 126 native arm64 PyTorch fits, 25,200 optimizer updates,
126 frozen score directions and paired location contrasts. Four compute
threads, zero data workers, checkpoints every 100 updates and PID
heartbeats; twelve focused tests passed. Repeat `freeze` to resume.
`cached_verified`: existing proposal and donor forecast packets,
recording-level evaluation terms and exact source resamples. The
weights and scoring packets stay local.

[Protocol](protocol.md) | [Freeze](freeze.json) | [Readout](readout.json) |
[Paired score contrasts](paired_contrasts.json).
