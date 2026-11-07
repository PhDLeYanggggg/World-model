# Pair Supervision Improves Observed Geometry, Not Conservative Utility

Six new native arm64 PyTorch fits isolate a relative-path loss.
They use 69,746 rows from nine nonempty fitting sources and exclude
all 14 exposed development sources. The fitting graph has 176,728
edges across 1,176 queries, of which 169,530 have at least one
co-observed future step. Edges are repeated windows, not unique
agent pairs. Every edge is constructed from current positions
within one query before using labels for the training loss.

Within each seed, both arms share frozen backbones, initialization,
900 updates, and identical draws of 64 individual rows plus the
two endpoints of 32 edges per batch. Both optimize individual ADE
over all 128 drawn rows; only one adds 0.25 times relative-path
error on co-observed future steps. Both keep the same missing-label
masks and normalization. Training took 16.6 seconds with two
Torch threads and no loader workers. The six forecasts were
frozen before scoring development outcomes.

| Treatment vs individual-only, same recipients | Effect | Paired 95% CI |
| --- | ---: | ---: |
| Observed relative-path ADE reduction (px) | +0.044 | [+0.022, +0.069] |
| Observed terminal relative-path error reduction (px) | +0.368 | [+0.062, +0.874] |
| Relative-path full-grid lower contrast (px) | -0.239 | [-0.607, -0.022] |
| Relative-path terminal lower contrast (px) | -0.533 | [-1.324, -0.058] |
| Individual available-label utility (pp) | +0.032 | [-0.012, +0.091] |
| Individual conservative utility (pp) | -0.107 | [-0.320, +0.018] |

All contrasts hold fixed the 9,675 original switch recipients
and per-query counts. Three seeds are averaged within each
source before 3,000 paired resamples of 14 sources. The observed
and full-grid pair measures differ in support and denominator:
the negative lower contrast does not prove that every possible
missing-label completion loses. Relative-path error in image
pixels is not a physical safety or collision measure.

The original gain control, individual-only and pair-trained
arms pass 4/5/5 of 102 recording-seed views. The two trained
arms have the same 81 all-risk exceedances and 23 undefined
easy views. Worst signed easy degradation moves from 65.52%
to 71.06% when adding the pair loss; both exceed 2%. The
pair-trained candidate's conservative utility against original
gain is -0.819 pp [-2.050, +0.019]. The observed geometry
benefit does not meet the registered joint criterion and does
not change deployment. It does not rule out other relative
objectives or establish benefit under an independent split.

[Protocol](protocol.md) | [Registration](registration.json) |
[Training receipt](train_execution.json) | [Forecast freeze](infer_execution.json) |
[Analysis](analysis.json) | [Score](score_execution.json) |
[Replay](replay_execution.json).

```sh
.venv-pytorch/bin/python scripts/run_m3w_relative_path_training.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_relative_path_training.py infer
.venv-pytorch/bin/python scripts/run_m3w_relative_path_training.py score
.venv-pytorch/bin/python scripts/run_m3w_relative_path_training.py replay
```

Nine focused tests passed; score replay matches the original
scalar artifact exactly. Atomic checkpoints are private. This
is exposed development only, in image pixels and eight/twelve
annotation steps, not metric distance or seconds. No external
access, independent confirmation, Stage5C execution or SMC.
