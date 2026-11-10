# OpenDD: matching the donor path family is not enough

The original additive policy selects learned forecast paths for
98.34% of its recipient slots. I replayed the original donor draws
and changed only the training path family, substituting a learned
path whenever it was legal. Donor, query, recipients, peak radius,
model, loss, optimizer, and 200-update budget stay fixed. The angle
nominator is the same source-excluded model. Held nominees and
all four prior controls reproduce exactly.

On the same 66,555 repeated training examples, the learned-family
share moves from **34.84% to 96.90%**. The path changes in 56,904
examples, and the exact actor/angle index changes in 33,393.
Strict positives move from **1,449 (2.18%)** to **2,659 (4.00%)**,
nearer to the fixed outer menu's 420/8,874 (4.73%). This alignment
does not match the incumbent's method of choosing actors.

## Held actions at the same budget

| Screen | Strict / 526 | Risk / 526 | Strict / 1,368 | Nominee AUROC |
| --- | ---: | ---: | ---: | ---: |
| Original action margin | 54 | 261 | 107 | 0.516 |
| Uniform pointwise | 41 | 279 | 105 | 0.605 |
| Aligned pointwise | 35 | 273 | 99 | 0.617 |
| Uniform budget rank | 40 | 277 | 96 | 0.549 |
| Aligned budget rank | 44 | 296 | 107 | 0.579 |

At 526 rotations, aligning the budget ranker adds four strict
passes in the pooled count. The conservative trajectory contrast
against the uniform counterpart is **-0.000003 pp**
[-0.000019, +0.000011]. Its relative-motion contrast is
**-0.000382** native units [-0.000695, -0.000145]. Both pointwise
and budget ranking remain below original margin on conservative
trajectory utility at 526 rotations; the latter is **-0.000111
pp** [-0.000230, -0.000021].

At the rounded 15% quota, the aligned budget ranker ties original
margin's raw strict count. It also adds 11 passes over the uniform
budget ranker, but loses conservative trajectory utility against it
by -0.000057 pp [-0.000112, -0.000010]. Passing counts and the
magnitude of paired gains are different outcomes. Intervals use
3,000 draws over seven location means, averaging seeds and
directions first. The 8,874 views repeat 493 recordings.

| Separately calibrated screen | Rotations | Original risk passes | Risk, both references and positive incremental gain |
| --- | ---: | ---: | ---: |
| Original action margin | 94 | 37 | 11 |
| Uniform pointwise | 28 | 14 | 1 |
| Aligned pointwise | 21 | 15 | 7 |
| Uniform budget rank | 25 | 17 | 7 |
| Aligned budget rank | 12 | 6 | 2 |

There is no release change. Matching the path family and increasing
positive training labels does not solve conservative utility or
relative-motion selection. The next controlled change is to generate
donor anchors with a source-excluded version of the actual additive
recipient chooser, rather than uniformly choosing actors. Keep
the held action bank and thresholds' training roles unchanged.
No new external confirmation was opened.

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_vehicle_family_alignment.py freeze
.venv-pytorch/bin/python scripts/run_m3w_vehicle_family_alignment.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_vehicle_family_alignment.py
```

`fresh_run`: 126 native arm64 Torch fits, 25,200 updates across
three seeds, 126 direction scores and paired location intervals.
Checkpoint every 100 steps, with saved optimizer, RNG and heartbeat.
The freeze was pushed as `134a09184` before held scoring.
`cached_verified`: original nested teachers, uniform controls,
held nominees and cohort. The targeted regression suite passed
148 tests. This is OpenDD development evidence, not a metric or
independent external result.

[Protocol](protocol.md) | [Pre-score freeze](freeze.json) | [Readout](readout.json).
