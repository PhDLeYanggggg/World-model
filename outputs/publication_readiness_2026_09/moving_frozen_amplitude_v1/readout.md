# Count-Preserving Forecast Amplitude Control

**Result: shrinking interventions improves recording risk at fixed coverage,
but does not solve missing support.**
This experiment changes only the trajectory offered at each frozen switch.
It uses the same 34 exposed recordings, 14 source groups, three forecast
seeds, CV reference, reference cuts, and 6,952 switches in every arm.
No model was fitted or selected from these results.

| Amplitude | Recording passes | Total risk >2% | Signed easy >2% | Undefined easy reference |
|---|---:|---:|---:|---:|
| Original | 2/102 | 75 | 8 | 35 |
| Half | 4/102 | 62 | 0 | 35 |
| Quarter (registered primary) | 15/102 | 39 | 0 | 35 |
| Eighth | 30/102 | 11 | 0 | 35 |
| Past-motion cap | 2/102 | 75 | 6 | 35 |

The primary quarter-versus-full improvement is **+12.70 pp** in
source-averaged recording pass rate (paired 3,000-draw source
bootstrap 95% interval **[+2.38, +24.60] pp**). The quarter arm's
paired completion-lower utility against CV is **+0.057 pp**
([+0.018, +0.095]). Its lower contrast against the original
amplitude is **-0.056 pp** ([-0.140, +0.035]); a utility improvement
over the unshrunk policy is not established. The eighth and half
arms are sensitivity checks, not a posthoc choice of operating point.

## What the Fixed Roster Reveals

Of the original 75 total-risk violations, 74 already exceed 2%
with the observed positive-harm numerator and the original
selected reference denominator. One crosses only after adding the
wholly unknown disagreement radius. Eighteen views have no total
reference. This decomposition is within the original observed
denominator contract, not a claim about the unavailable complete
trajectory error.

On an observed point, convex Euclidean error at the interpolated
forecast is at most the same interpolation of the two errors.
Consequently, the observed excess and its positive part have
rowwise alpha-scaled upper bounds. Wholly unknown disagreement
scales exactly, while the selected reference denominator does not
change. The numeric checks pass in every view for both uniform
and rowwise coefficients. For the quarter arm the convex bound
alone suffices in four views, versus 15 actual passes after
scoring: the sufficient bound is not an equality. It cannot
recover missing reference or certify physical safety.

The radius cap based on past motion barely changes risk here.
Calibrating amplitude is a more direct control than re-ranking
the same actions, but the observed benefit is small in error
units and confined to development. Missing-reference acquisition
is a separate task. No arm meets the complete original gate.

## Reproduction

Registered in `a424f53c` before scoring:

```sh
.venv-pytorch/bin/python scripts/run_m3w_frozen_amplitude.py score
.venv-pytorch/bin/python scripts/run_m3w_frozen_amplitude.py replay
```

Each pass takes about 11 seconds locally. Scalar replay is exact.
No threshold was tuned or deployed; no external label, metric or
seconds-level claim, Stage5C or SMC.
