# A better forecast did not make a better rotation decision

I repeated the one-rotation comparison after noticing that the old
distance weights used actor-centered last positions. All those last
positions are zero. The corrected rule uses the checked shared-frame
offsets. It still has to use the same two actors, the same query,
the same displacement at every future step, and the original 2% checks.
The old centered rule reproduces exactly.

| Choice rule | Rotations | Risk passes | Risk and both references |
| --- | ---: | ---: | ---: |
| Original additive, no rotation | 0 | 4,379 | 2,975 |
| Old proxy, centered coordinates | 364 | 4,369 | 2,962 |
| Same proxy, actual spacing | 363 | 4,369 | 2,962 |
| Neural forecast, ego only | 4,042 | 4,353 | 2,886 |
| Neural forecast, two observed neighbors | 5,162 | 4,301 | 2,838 |
| Ego-only angle at the exact same rotating actors | 5,162 | 4,373 | 2,907 |

There are 8,874 repeated views over 493 distinct recordings. The
3,730 undefined risk views do not become passes. Available-step
ADE, on the same recording rosters, is 6.14181 for CV, 4.98500
for the ego-only model and 4.99235 for the neighbor model in native
coordinates. These are direct forecast scores, before choosing any
angle. The neighbor model does not improve the ego-only ADE
(ego minus neighbor: -0.00735, 95% location interval
[-0.03272, +0.02127]). Both learned forecasts lower this ADE relative
to CV by about 19%, but both induce many more rotations than the
three-reference proxy, with fewer dual passes than the original.

| Paired contrast | Conservative trajectory gain (pp) | Conservative relative gain (native units) |
| --- | ---: | ---: |
| Actual spacing minus centered proxy | -0.000048 [-0.000103, -0.000005] | -0.000311 [-0.000541, -0.000081] |
| Neighbor minus ego, same scoring rule | -0.001569 [-0.002746, -0.000754] | -0.009151 [-0.013556, -0.005496] |
| Neighbor minus ego individual score, exact same actors | -0.001275 [-0.002282, -0.000579] | -0.006779 [-0.010492, -0.003758] |

The second contrast isolates the input change at a fixed architecture
and scoring rule; its intervention rates differ. The third fixes the
intervention mask and recipient, but also changes how angles are
ranked. It is not a pure feature ablation. Intervals resample seven
locations 3,000 times, after averaging directions and seeds. Relative
trajectory comparisons are not claims about physical safety.

The useful result is the mismatch between forecast accuracy and the
benefit of editing. I would keep the original output rather than
replace it with these angles. The next comparison should train on
the loss **difference for a legal edit versus keeping its original
direction** and compare against fixed-angle choices at the same
recipients. Better average trajectory prediction is not the target.

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_vehicle_direction_context.py freeze
.venv-pytorch/bin/python scripts/run_m3w_vehicle_direction_context.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_vehicle_direction_context.py
```

`fresh_run`: 126 native arm64 Torch fits, 25,200 updates and 126
development direction readouts. Three seeds, four compute threads,
zero loader workers. Checkpoints every 50 updates, saved sampling
state, and PID heartbeats. A test reproduces uninterrupted weights
after a checkpoint restart. The expanded regression suite passed
137 vehicle, angle, Songdo and lineage tests.
`cached_verified`: source cohorts,
shared-frame offsets and original proposals. The pre-score freeze
was committed and pushed as `447f80650` before the held readout.
No external labels were opened; no policy was promoted.

[Protocol](protocol.md) | [Freeze](freeze.json) | [Readout and intervals](readout.json).
