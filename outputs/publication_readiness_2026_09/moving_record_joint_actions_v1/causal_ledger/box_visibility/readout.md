# Observed Box Geometry for Insertion Retrieval

**Result: two feasible probes from one recording, no frozen-policy gain.**
The feature build joins 27 observed box descriptors to all 125,540
rows from 34 recordings. It checks released annotation hashes and
exact box-center equality against all eight observed trajectory
steps. Box overlap is a geometry proxy, not an occlusion label.

| Arm | First-query easy-reference hits | First-query non-loss rescues | Folds with a cutoff | Insertions | Recording passes |
|---|---:|---:|---:|---:|---:|
| Original direct | not probed | not probed | not applicable | 0 | 2/102 |
| Trajectory and candidate features | 17/102 | 0/102 | 1/42 | 4 | 2/102 |
| Same features plus observed boxes | 17/102 | 2/102 | 1/42 | 1 | 2/102 |

The first-query rescue contrast is +0.95 pp (paired 3,000-draw
source bootstrap 95% interval [0.00, +2.86] pp). Both verified
probe rescues came from the same recording and source, repeated
under two forecast seeds. Neither fold had a donor cutoff, so
neither insertion was taken by the operating policy. Reference
retrieval changed in both directions: five views gained a hit
and five lost one. The missing-reference subgroup moves from
eight to nine first-query support hits, without a whole-policy gain.

Frozen paired completion-lower utility for boxes versus trajectory
is +0.00195 pp (95% interval [-0.00014, +0.00599]).
The box policy still leaves 35 undefined references and eight
signed-easy violations. These results do not establish either a
cross-recording lift or a safe upgrade.

## Reproduction

Registered in `2814a1d3` before feature building and fitting.
With verified private inputs:

```sh
.venv-pytorch/bin/python scripts/run_m3w_box_visibility.py build
.venv-pytorch/bin/python scripts/run_m3w_box_visibility.py fit
.venv-pytorch/bin/python scripts/run_m3w_box_visibility.py score
.venv-pytorch/bin/python scripts/run_m3w_box_visibility.py replay
.venv-pytorch/bin/python scripts/audit_m3w_box_visibility.py
.venv-pytorch/bin/python scripts/build_m3w_insertion_appendix.py
```

All 84 native arm64 Torch fits completed 220 updates in about
86 seconds, with four compute threads, zero loader workers and
local checkpoints. Replay agrees exactly. The 102 views are
34 recordings across three frozen forecast seeds and 14 source
groups, not independent samples. The original 2% and support
rules remain unchanged. No deployment or independent test,
metric, seconds-level, Stage5C or SMC claim follows.
