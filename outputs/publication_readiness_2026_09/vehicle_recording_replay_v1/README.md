# OpenDD vehicle recording replay

I built a standalone replay for the [two-edit ranking experiment](../opendd_vehicle_anchor_rank_v1/README.md).
It starts one step earlier than the previous reports: the inputs are
observed error changes, missing-step disagreement, selected easy
reference mass and the other recording-level risk components. It
contains no coordinates, weights, source names or saved pass flags.
It recomputes the original risk checks and the calibration score
prefix on the designated calibration side of each direction.

The replay matches all 126 directions, 493 distinct recordings, and
8,874 repeated views from three seeds. It matches the original
threshold and calibration count in every direction and reproduces
the released counts and paired location intervals.

| Frozen result | CV/OLS4-only head | Relative-target head |
| --- | ---: | ---: |
| Held matched proposals satisfying the full condition | 373 | 408 |
| Directions with a nonempty calibration prefix | 8 | 9 |
| Held replacements admitted after calibration | 3 | 14 |
| Admitted with undefined easy reference | 2 | 7 |
| Admitted passing original risk | 0 | 6 |
| Admitted passing original risk, both baselines and the additive contrast | 0 | 1 |

## Does a higher score identify better replacements?

I compare the five highest scores in each held direction with uniform
sampling of five from that **same** direction's finite-score proposals.
The control uses its exact expected pass rate. Ties are averaged. These
are diagnostics on frozen candidates, not another release policy.

| Top five minus same-direction uniform, pp | CV/OLS4-only | Relative-target |
| --- | ---: | ---: |
| Selected easy reference available | +1.25 [-1.91, +3.90] | -1.10 [-5.19, +3.08] |
| All original risk checks pass | +1.30 [-0.97, +3.26] | -4.23 [-7.35, -0.96] |
| Also beats CV and OLS4 | +4.53 [+1.66, +7.71] | -0.03 [-2.96, +2.71] |
| Also no conservative trajectory loss and better relative motion vs additive | +0.07 [-2.33, +2.63] | -0.25 [-2.01, +1.79] |

The relative head finds more useful matched proposals than the
same-capacity control, but its high scores do not identify the full
condition and are worse on the risk checks. The CV/OLS4-only head
ranks dual-reference passes better, not the full joint condition.
The missing easy reference accounts for seven of the relative head's
eight admitted risk failures. The result points to the release score
and support, not the absence of legal actions. I retain the CV fallback.

The diagnostic uses the existing 3,000 resamples over seven equal-weight
location means, after averaging seeds and calibration choices within
location. It does not tune a threshold or establish out-of-domain risk
control. No independent outcomes, deployment or manuscript were changed.

## Replay

The archive is local at
`data/stage_cvpr2027_experiments/submission_admin/vehicle_recording_replay.zip`.
Extract it and run `python -I -B -S verify.py`. It runs with the standard
library alone and checks all archive member hashes. The previous
anonymous archive for the main claims remains unchanged.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/package_m3w_vehicle_recording_replay.py
.venv-pytorch/bin/python scripts/audit_m3w_vehicle_recording_ranking.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_vehicle_recording_replay.py tests/test_m3w_vehicle_anchor_rank.py tests/test_m3w_vehicle_relative_ceiling.py tests/test_m3w_reviewer_evidence.py
```

`fresh_run`: recording-component export, standard-library replay and
paired ranking diagnostic. `cached_verified`: frozen OpenDD packets,
data cohorts and scores. Nineteen focused tests passed. The archive
does not reproduce tracking, model training or upstream inference.

[Replay receipt](receipt.json) | [Rank-score contrasts](ranking_diagnostic.json).
