# OpenDD: One Missing Label for All Three Training Seeds

**2026-11-08.** Recomputed the frozen context comparison on seven exposed
OpenDD locations and three seeds, without fitting any heads. The original
per-location/per-seed lower scores reproduced. This is a follow-up
sensitivity analysis, not an independent policy test.

## What Changed

The original bound couples the same seed across two policies before
applying the reverse triangle inequality at an unobserved target-step.
But all seeds predict the **same** unknown coordinate. Permuting the
three alternative forecasts leaves the sum of their errors unchanged.
The minimum-cost of the six assignments therefore gives a bound no
wider than the original one. This is an equal-weight
[transport bound](https://research.google/pubs/computational-optimal-transport/),
not an assertion that the distance-to-point extrema attain that bound.
The observed label term, equal site averaging and original observed CV
normalization stay fixed.

To check attainability of either sign, each incomplete target row uses
the missing coordinates from one entire frozen path template: CV or any
seed from either of the two compared policies. The known steps never
change. The two reported template values select the lower/higher
contribution across that finite library per row, then average over
locations. Each template completion is shared across seeds. Extremal
completions can differ by comparison. These are admissible coordinate
completions for this sensitivity, not imputed or physically validated
trajectories.

## Results

| Time-aligned motor vs. | Observed term, pp | Original diagonal interval, pp | Seed-shared interval, pp | Attainable template values, pp |
|---|---:|---:|---:|---:|
| Fixed half | +0.003996 | [+0.002947, +0.005045] | [+0.002951, +0.005041] | [+0.002970, +0.004989] |
| Pedestrian context | +0.001520 | [-0.000702, +0.003741] | [-0.000615, +0.003654] | [-0.000503, +0.003505] |
| Shuffled motor | +0.000533 | [-0.001233, +0.002298] | [-0.001194, +0.002260] | [-0.001081, +0.002148] |

Across 4,489 unique missing target-step occurrences on the nonoverlapping
roster, the assignment tightens 16, 119 and 100 steps for the three
respective contrasts. Neither direct matched-context sign is identified:
the observed term is positive but actual template completions reverse
both point comparisons. The separate location-bootstrap 95% intervals
for the shared lower bound are [-0.003114, +0.002239] against pedestrian
context and [-0.003477, +0.000842] against shuffled motor. The bootstrap
resamples seven sites after averaging seeds, never target steps.
Outer intervals are not confidence intervals, and conditional observed
mass is not the unknown full-grid CV error.

**Conclusion:** An overly loose per-seed bound is not enough to explain
the missing evidence for actor-specific motor information. A small
positive contrast to fixed half is still possible without proving
improvement over the matched information controls. The context result
remains a development null for the registered contribution test; this
does not establish harm or equivalence. No recording-risk decision is
changed and no independent outcomes are opened.

## Replay

~~~sh
.venv-pytorch/bin/python scripts/audit_m3w_opendd_seed_shared_completion.py register
.venv-pytorch/bin/python scripts/audit_m3w_opendd_seed_shared_completion.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_seed_shared_completion.py tests/test_m3w_shared_pair_completion.py tests/test_m3w_opendd_location.py tests/test_m3w_exchange_context.py tests/test_m3w_motion_centered_completion.py
~~~

The script binds the frozen heads and parent replay, and cannot fit new
heads. An initial parent comparison-name mismatch stopped before
producing an analysis; the name mapping was corrected without changing
any comparison. Two complete scores reproduced exactly, 12 focused
tests passed, and the DroneCrowd role manifest verified separately.
[Registration](registration.json) | [Per-site evidence](analysis.json) |
[Paper and supplement](../evidence_manuscript_v36/README.md).
