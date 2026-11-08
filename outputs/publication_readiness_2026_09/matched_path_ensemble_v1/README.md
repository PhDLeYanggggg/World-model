# Does Matching Candidate Averaging Repair Transfer?

The earlier OpenDD-to-TUMDOT selector learned on individual donor
forecasts but inferred on seven-model averages. I tested that mismatch
directly. Each new OpenDD fitting row uses seven producers that excluded
its location: one single-location exclusion and six pair exclusions.
Each candidate family reuses the same location-excluded refiner across
those seven producers. Thus the refiner weights and every source row's
intervention recipient and peak radius are fixed. This is not a perfect
replica of target ensembles: the target uses seven distinct refiners, and
the source members have fewer training locations. The common TUMDOT
test-time path menu does not change at all.

Six native Torch fits, three seeds and 3,000 updates; identical initial
seeds, sampling, architecture, loss and selection rules to the single
producer study. All six old policies replay exactly across the nine
already exposed recordings at both radii. Target labels were not used
for training or selection, and the separate confirmation set was not
accessed. This is a paired development diagnostic, not a new test set.

| Half-radius agreement policy | Lower vs CV (pp) | Direct vs matched hindcast (pp) | 2% passes |
|---|---:|---:|---:|
| Single producer, full history | -0.0618 | -0.1661 | 21/27 |
| Ensemble, full history | -0.0682 | -0.1674 | 21/27 |
| Single producer, action only | -0.0228 | -0.1052 | 22/27 |
| Ensemble, action only | -0.0147 | -0.0925 | 21/27 |

Source averaging changes the hindsight best arm for 4,110 of 30,073
source row-seed cases with a strict original winner (13.7%), and moves
paths by about 10.0% of their matched radius on average. On TUMDOT,
ensemble training changes the full-history choice on 488 of 4,368
row-seed cases (11.2%). Its **direct** conservative difference versus
the single-producer full head is **-0.0169 pp** (observed-only -0.0044).
Action-only agreement has a small direct gain **+0.0024 pp**, but loses
one recording-seed risk pass and remains below both strong references.
At the quarter radius both heads pass 27/27 and both still lose to
CV and matched hindcast. There is no independent-domain CI: the nine
recordings and three model seeds are from one campaign, not 27 sites.

**Decision.** This version of train-time averaging is not sufficient to
fix the selector. The negative result does not isolate the remaining
domain shift from differences in member training or refiner composition.
The useful next check is whether reliable causal short-prefix backtests
rank the candidate directions, before spending on larger gain heads.
Keep the original risk rule, the existing deployment and the DroneCrowd
confirmation seal. No Stage5C or SMC.

```sh
.venv-pytorch/bin/python scripts/run_m3w_matched_path_ensemble.py
.venv-pytorch/bin/python scripts/run_m3w_matched_path_ensemble.py --replay
```

[Frozen protocol](registration.json) | [Fitting and action shifts](head_freeze.json) |
[All evaluations](analysis.json).
