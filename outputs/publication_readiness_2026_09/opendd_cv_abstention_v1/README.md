# Can a CV-Anchored Selector Learn When to Abstain?

**The easy slice was preserved, but positive gain over CV was not established.**
I reused the hash-verified OpenDD cohort: 10,475 pedestrian targets in 3,395
non-overlapping queries. This is development on an already examined source,
not a new external confirmation. The 30 DroneCrowd confirmation groups were
not accessed. Deployment and the original 2% rule are unchanged.

## Method and Test

I replaced OLS4 with last-displacement CV as the default prediction and let
the model select OLS4 or one of the four previous past-only alternatives.
The CV-only and dual-reference models have the same size, causal features,
update counts and draws. The second model learns to penalize harm relative
to OLS4 as well as CV. The nested folds exclude the outer location from both
training and threshold selection. Thresholds are selected on six other sites
using predictions from five-site fits. Each fold and seed also has a fixed
one-edit comparison with matching queries and candidate trajectories.

The complete experiment used **168 fresh Torch fits**: 126 inner fits shared
across directions of 21 excluded pairs, and 42 outer fits. There were three
seeds and 300 updates per fit. Native arm64 CPU, four compute threads, no
workers. The fresh run took 30.69 seconds with verified input caches; readout
from the saved checkpoints matched the complete analysis byte for byte.

## Result

Gain is the conservative paired difference over all scheduled steps, divided
by observed CV error mass, averaged equally over the seven sites after seed
averaging. Intervals are paired site bootstrap (3,000 draws). Switch totals
sum over the three seeds; the seven locations are not independent cities.
Easy values show the worst signed change across site-seed views for the
descriptive CV-easy slice, not a formal per-view safety result.

| Policy | Gain vs CV, pp [95% interval] | Switches | Worst easy ADE change |
|---|---:|---:|---:|
| CV fallback | 0 | 0 | 0% |
| CV-only forced | -1.052 [-2.827, +1.852] | 10,122 | +19.260% |
| Dual-reference forced | -1.263 [-2.409, +0.571] | 10,122 | +17.545% |
| CV-only calibrated | -0.102 [-0.288, +0.006] | 399 | +1.132% |
| Dual-reference calibrated | +0.010 [-0.001, +0.030] | 44 | +0.068% |

The inner selection found a threshold in 11 of 21 folds per arm; it chose
unconditional CV in the other 10. On the outer views, the calibrated dual
arm edited only **0.43%** of forced-edit opportunities. Relative to the
calibrated CV-only arm its paired gain was +0.067 pp [+0.005, +0.164], but
they did not switch the same number of times. At matched forced counts,
the corresponding difference was -0.390 pp [-1.510, +0.403]. Thus the
calibrated comparison mainly supports being more selective; it does not
isolate better recipient or candidate ranking. On its 44 exact edited
queries the dual policy did not establish a gain over hindcast (-0.072 pp
[-0.206, +0.003]). Observed dual ADE/FDE were 0.60822/1.35326, essentially
the same as CV's 0.60830/1.35341 in publisher coordinate units.

## Why the Result Matters

A separate **post-hoc hindsight** readout chooses at most one edit per query
only if its full-grid lower-bound gain is positive, and checks that no known
easy target incurs harm. It finds **+17.013 pp** [+14.682, +19.212] with
3,204 edits on one pass and zero positive harm on known easy rows. It uses
future outcomes to make choices; this is neither a model nor evidence of
safety for unknown labels. In a one-recipient query, a positive lower-bound
gain on a fully observed easy target already implies zero ADE harm to that
target. The result shows why searching additional thresholds on the same
score is not the most informative next step: the forecast menu has useful
options in hindsight, but the causal gain ranker cannot reliably identify
them. It does not establish that the gap is predictable from past inputs.

Keep CV for this diagnostic. Test a gain-ranking objective against both
references with matched candidate and switch counts before another policy
search. Any method changed using OpenDD results needs fresh evidence before
claiming generalization. No multimodal or joint-selection claim follows
from this experiment. Stage5C and SMC remain disabled.

```sh
# Build the original hash-bound cohort first if it is not already available.
.venv-pytorch/bin/python scripts/run_m3w_opendd_location.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_cv_abstention.py
.venv-pytorch/bin/python scripts/report_m3w_opendd_cv_abstention.py
```

The runner records PID and progress, takes a single-process lock and resumes
matching checkpoints under `data/stage_cvpr2027_experiments/opendd_cv_abstention_v1/`.
The full threshold table is generated as `analysis.json`; only the compact
readout is shared, with the full table's SHA-256. No tracks, images, caches
or weights are distributed.

[Registered protocol](protocol.md) | [Registration](registration.json) | [Compact metrics](metrics.json)
