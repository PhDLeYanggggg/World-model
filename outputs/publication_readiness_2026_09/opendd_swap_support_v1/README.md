# OpenDD: better swap ranking, but not a better policy

I separated support (the fraction of the forecast-change bound on
observed steps) from gain and harm on those steps. When support is
zero, the conditional outcome has no training loss. The lower-bound
prediction restores the missing-step debit rather than imputing
unobserved positions. A separate head predicts whether that lower
endpoint is positive. The footprint variant adds only the occupied
rectangle of the eight past observations and its relation to CV
rollouts. This is not a camera boundary or scale calibration.

## Ranking at the same nominations

| Score | Individual ROC AUC | Relative ROC AUC | Relative average precision |
| --- | ---: | ---: | ---: |
| Archived local gain (v194) | 0.585 | -- | -- |
| Fused expected effect (v196) | 0.556 | 0.569 | 0.116 |
| Direct sign, no footprint | 0.611 | 0.647 | 0.147 |
| Direct sign, with footprint | **0.679** | **0.683** | **0.160** |

On the same 8,814 legal nominations, the footprint sign score gains
**+0.114 relative AUC [0.104, 0.125]** against v196 and **+0.093
individual AUC [0.059, 0.133]** against v194. At identical objectives,
adding the footprint gains **+0.035 [0.020, 0.053]** in relative AUC
over the no-footprint ablation. The relative support MAE falls from
0.146 to 0.122 and individual support MAE from 0.122 to 0.106.
Reconstructing the expected lower endpoint rather than directly
ranking its sign gives relative AUC 0.601. The intervals use 3,000
paired resamples of seven locations after averaging the three seeds.

## The ranking gain does not survive allocation

The preregistered ranking screen passed. I then froze a separate
[allocation comparison](../opendd_swap_support_allocation_v1/protocol.md)
using the same nominations, 353 swaps, and the original per-direction
95--100% movement range. There was no additional fit or coefficient
search. The joint score is individual sign probability plus half the
relative sign probability.

| Rule | Qualifying swaps / 353 | Risk / dual passes over 8,874 repeated views |
| --- | ---: | ---: |
| Archived opportunity | 68 | 4,362 / 2,984 |
| Individual sign | 55 | 4,363 / 2,966 |
| Combined sign | 54 | 4,361 / 2,964 |

The combined rule's conservative trajectory gain versus opportunity
is **−0.000150 pp [−0.000293, −0.000014]**. Compared with individual
sign, the combined rule's observed relative contribution has an
interval spanning zero. Its conservative relative gain is **−0.000272
[−0.000579, −0.000073]** native units; its full-grid missing-label
point bounds are **[−0.000272, +0.000243]**. These bounds do not
identify the full-grid sign. The worst signed easy degradation is
17.67% for all three rules and the frozen additive baseline. No
deployment change.

The improvement is in ranking the sign of individual outcomes, not in
allocating a fixed number of changes with joint constraints. The next
test should allow abstention and use a signed, magnitude-sensitive
score that is calibrated on a disjoint development group. It should
retain the existing 2% checks rather than select a new threshold on
these same held directions. This differs from imputing missing history
or adapting a model at test time, as in [T4P](https://openaccess.thecvf.com/content/CVPR2024/html/Park_T4P_Test-Time_Training_of_Trajectory_Prediction_via_Masked_Autoencoder_and_CVPR_2024_paper.html).
The 8,874 allocation views repeat 493 distinct recordings. This study
does not add an independent test or support a metric or seconds claim.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_swap_support.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_swap_support.py score
.venv-pytorch/bin/python scripts/run_m3w_opendd_swap_support_allocation.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_swap_support_allocation.py score
```

`fresh_run`: 42 arm64 CPU fits and 12,600 updates (21.9 s), ranking
evaluation (7.9 s), and held allocation (72.2 s). Four Torch threads,
zero workers, 50-step checkpoints and PID progress logs. Ranking and
allocation predictions were published before their respective scoring.
`cached_verified`: the previous OpenDD packets and 126 directions;
replay reproduces the readouts without new optimizer updates.
183 related tests passed. No CREATE job, Stage5C, SMC, or new external
outcome access.

[Ranking protocol](protocol.md) | [Ranking results](readout.json) |
[Allocation results](../opendd_swap_support_allocation_v1/readout.json) |
[v197 draft](../evidence_manuscript_v197_swap_support/README.md)
