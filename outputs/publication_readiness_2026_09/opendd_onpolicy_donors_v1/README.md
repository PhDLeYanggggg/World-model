# Train on policy-matched allocations

The training allocation generator now matches the algorithm used to produce
the evaluated incumbent: learned recording counts followed by the score-only
count/movement-constrained optimizer. The selector features, architecture,
updates and evaluated action mask are unchanged.

## Result

Policy matching improves the paired optional point estimate over the previous
selector, but does not beat keeping the incumbent. The registered primary
comparison fails. No new policy is deployed.

| Policy | Allocation exchanges / 101 eligible views | Full-grid passes / 606 views |
| --- | ---: | ---: |
| Keep incumbent | 0 | 55 |
| Previous paired optional | 94 | 47 |
| New pointwise forced | 101 | 42 |
| New pointwise optional | 83 | 45 |
| New paired forced | 101 | 50 |
| New paired optional | 82 | 52 |
| Uniform exchange on the new paired switch mask | 82 | 49.1784 expected |
| Same-mask hindsight oracle including keep | diagnostic | 100 |

All strategies use the same 780 forecast edits. Allocation exchanges are not
additional forecast edits. Pass counts are not FDE improvements. With 3,000
paired-location bootstrap draws after averaging seeds within each location:

| New paired optional minus | Equal-location difference (pp) | 95% CI (pp) |
| --- | ---: | ---: |
| Incumbent | -0.1125 | [-0.3375, 0.0000] |
| Own-mask uniform | +0.0891 | [-0.0775, +0.3449] |
| Previous paired optional | +1.1024 | [0.0000, +3.0071] |

The primary requires all three lower bounds to be strictly positive. None
meets that requirement. There are seven development locations, 202 recordings
and 606 recording-seed views, not 606 independent scenes. Original-risk
undefined views remain undefined: 399 for paired optional versus 397 for
incumbent. Observed risk-violation counts are 121 for each; this is not a
release certificate for all views.

| Seed | Incumbent | Previous paired optional | New paired optional |
| --- | ---: | ---: | ---: |
| 17 | 19 | 17 | 17 |
| 29 | 17 | 13 | 20 |
| 43 | 19 | 17 | 15 |

By location, incumbent to new paired full-grid counts are rdb1 43 to 40,
rdb2 0 to 0, rdb3 3 to 3, rdb4 0 to 0, rdb5 1 to 1, rdb6 6 to 6 and
rdb7 2 to 2. The improvement over the previous selector comes from rdb1
(36 to 40) and rdb3 (2 to 3), not broad gains over the incumbent.

## What changed and what still fails

The new paired selector rescues 10 cases versus the previous 5, but still
destroys 13 incumbent passes. It keeps 19/101 allocations instead of 7/101.
Its remaining 48 missed feasible views are evidence of selection error, not
proof that their oracle outcomes are attainable by a deployable model.

Donor generation produces 1,786 repeated slates and 26,028 options, compared
with 4,507 and 67,275 previously. The treatment deliberately changes the
generator, so equal optimizer updates do not mean equal training examples.
Anchor-positive frequency moves from 10.49% to 15.06%, still below 26.73%
in the fixed eligible held-location views. New donor slates include 791
all-negative, 14 all-positive and 981 heterogeneous cases.

The fitted model does learn the donor task. In-sample paired choices produce
807 original-dual passes from 269 passing anchors, with 548 rescues and 10
losses across 1,786 repeated training slates. Predicted selected-pass mean is
44.73%, observed is 45.18%, and ten-bin ECE is 1.80%. On the 101 eligible
held-location views, its selected-pass mean is 67.20%, observed is 20.79%,
and ECE is 52.87%. Brier score worsens from 0.1285 in-sample to 0.5158 held.
These are raw sigmoid diagnostics, not calibrated deployment probabilities.
They use the original-dual training label, distinct from full-grid counts.

Thus matching the allocation algorithm is insufficient. A substantial
generalization gap remains, together with seed sensitivity and a training
label that does not encode every full-grid condition. This experiment does
not isolate feature shift, model capacity and label mismatch as separate
causes. The donor/held difference is not evidence of future-label leakage;
the nested fit and cache lineage checks pass.

Next, assess the relative keep-versus-switch advantage on donor-held
validation locations before committing another outer choice set. Compare a
lower-capacity advantage estimator with the current head, and check whether
its ranking as well as its calibration transfers. Keep the same outer budget
and 2% rule. Do not use this readout to choose a new keep threshold, or open
external confirmation labels to rescue the result.

## Training and verification

105 triple-excluded nominee rankers, 63 double-excluded count heads and 42
incumbent-inclusive selection heads are trained. Total: 121,800 optimizer
updates. All 21 outer-location/seed choice sets were frozen. The choice freeze
and producer manifest were committed as `f9daea2d0` before outcome scoring;
registration was committed as `25c4c42b7` before training. These are new
Torch fits, not a NumPy substitute or reused weights presented as new fits.

The original 2% rule and empty-support treatment are unchanged. No external
confirmation labels or release policies have been modified. This is exposed
OpenDD development evidence, not independent confirmation.

## Reproduce

```bash
cd /Users/yangyue/Downloads/World
export LC_ALL=C OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_onpolicy_donors.py register
.venv-pytorch/bin/python scripts/run_m3w_opendd_onpolicy_donors.py producers
.venv-pytorch/bin/python scripts/run_m3w_opendd_onpolicy_donors.py fit
.venv-pytorch/bin/python scripts/run_m3w_opendd_onpolicy_donors.py score
.venv-pytorch/bin/python scripts/audit_m3w_onpolicy_donors.py
.venv-pytorch/bin/python scripts/diagnose_m3w_onpolicy_gap.py
```

Existing completed checkpoints resume with zero updates. The local caches and
weights live under `data/stage_cvpr2027_experiments/opendd_onpolicy_donors_v1/`
and are not included in Git. Upstream registered datasets, maps and checkpoints
must also be present. The registry binds their prior experiment lineage.

The pilot completed 4,000 real updates in 7.254 seconds with peak RSS 568 MB.
Completing the producer matrix added 101,000 updates in 172.859 seconds;
selector training took 108.443 seconds, peak RSS 1.67 GB. Local execution was
sufficient; no CREATE job was submitted. Every 100 updates saves optimizer,
weights and RNG. Per-invocation logs and the process lock are in the cache.

The zero-update selector replay completed all 21 cells in 16.882 seconds;
all choices reproduced exactly. Frozen scoring took 9.474 seconds. Nested
lineage verification checked 105 ranks, 63 count heads and 315 teacher caches.
Teacher-recording statuses include one unresolved optimization, excluded
rather than relabeled. Donor allocation has two numerical-check failures among
3,636 repeated recording views, also excluded as registered. There were 2,021
score-optimal and 1,613 zero-edit allocations; 1,848 had no legal exchange.
Repeated teacher/donor rows are not independent recordings. The 29 focused
tests cover the new role exclusions, budget
preservation, unresolved handling and zero-count aggregation alongside the
incumbent, listwise, risk and matched-exchange regressions; this is not a
whole-repository test claim.

See [the frozen protocol](protocol.md) for the primary comparisons and
[registration](registration.json), [producer freeze](producer_freeze.json)
and [choice freeze](head_freeze.json) for the bound inputs and outputs.

The [readout](readout.json), [lineage and replay audit](audit.json),
[train/held diagnostic](generalization_diagnostic.json) and
[runtime record](runtime.json) distinguish fitting, verified reuse and scoring.
No source weights or large caches are committed.

## Relation to existing work

Ross, Gordon and Bagnell (AISTATS 2011, introduction and Sections 2.1 and 3)
address policy-induced distribution shift with iterative data aggregation.
Here the decision is an offline allocation exchange under fixed count and
movement constraints. This experiment matches the donor-generating algorithm
once; it neither implements DAgger nor inherits its regret guarantees.
The potential contribution must come from demonstrated joint selection and
risk control, which this result does not yet establish.
[Original paper](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf).
