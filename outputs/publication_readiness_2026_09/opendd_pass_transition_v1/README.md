# Learning Recording Pass Transitions Directly

The [expanded support bank](../opendd_support_directions_v1/README.md)
contained 65 previously failing views with an exact feasible action.
Its composite advantage head never released on 55 of them. This
experiment keeps the bank, original gain, support priority and
native candidate radii unchanged, but replaces the training target.

Three binary outputs predict an unconditional rescue, a destroyed
joint pass, and loss of a defined easy reference. The head sees
only the candidate and observed prefix, never the original or
candidate outcome. To limit the influence of dense menus, training
samples a donor site, recording and candidate uniformly at each
level. The net-pass readout ranks by P(rescue) minus P(destruction).
The rescue-only readout drops the subtraction without refitting.
Both require a negative reference-loss logit and exceed the
90th percentile of donor-record maximum scores, floored at zero.
The threshold uses training data only and is not a risk guarantee.

| Policy | Edits | Joint pass | Risk pass | Undefined | Rescues | Destroyed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 943 | 138 | 172 | 295 | 0 | 0 |
| Frozen composite | 974 | 132 | 164 | 300 | 1 | 7 |
| Net pass | 963 | 133 | 167 | 298 | 5 | 10 |
| Rescue only | 967 | 136 | 169 | 297 | 6 | 8 |
| Composite, count-matched | 963 | 134 | 165 | 302 | 3 | 7 |
| Composite, same row and radius | 963 | 133 | 166 | 298 | 3 | 8 |

The count-matched comparator ranks each recording's best composite
candidate, then matches additions and replacements separately
within each site and seed. It is retrospective and sees the
available menu, not its outcomes. The same-row control holds
the exact net-pass query, recipient, radius, and count; only the
choice of direction uses the composite scores.

On the 606 development recording-seed views, net pass versus
count-matched composite is -0.00100 pp [-0.00180, -0.00020] in
paired conservative utility. The effect versus the original is
+0.00030 pp [-0.00085, +0.00176]. The exact row/radius contrast
is +0.00027 pp [-0.00015, +0.00100]. Intervals use 3,000 paired
location resamples after averaging three seeds. None establishes
a utility-and-risk improvement over the original.

## What the Training Target Did Not Fix

The net-pass head rescues four more views than frozen composite
but destroys three more original passes. It still never releases
in 56 of the 65 feasible new views. One location contains 45
of the 65 and scores zero rescues when held out. A second location
contains 19 and contributes four rescues. These are repeated
seed views, not independent opportunities. The concentration
motivates measuring candidate and transition support across
additional approved development sources before scaling the same
head or retuning its cutoff. In particular, test whether the
labels and candidate geometry transfer to the pedestrian development
protocol without using DroneCrowd confirmation outcomes.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_pass_transition.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_pass_transition.py --replay
```

`fresh_run`: 21 fits, 12,600 updates, native arm64 Torch, four
threads, zero loader workers, PID 81063, 23.2 seconds. Checkpoints
and heartbeat every 100 steps. A first attempt stopped before
updates on a label-shape test failure; the fixed three-column
target passed testing before fitting. `cached_verified`: frozen
forecasts, menu and composite head, and zero-update replay.
`not_run`: new CREATE training and independent confirmation.
Original 2% recording risk checks and undefined handling are
unchanged. No deployment change, submission, Stage5C or SMC.
