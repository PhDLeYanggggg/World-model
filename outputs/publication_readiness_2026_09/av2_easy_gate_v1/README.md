# AV2: Separating Easy Support Improves Risk Estimates, Not Matched Utility

The [independent selected-action veto](../av2_selected_veto_v1/README.md)
improved overall harm coverage but left the two easy terms
unresolved. I tested whether separating the existence of an
easy reference from harm conditional on that reference makes
the intervention decision more reliable.

## Same Choices, Different Easy Score

Let S indicate positive edited-easy reference mass. Let E
indicate that both original easy checks pass, including S.
The model learns P(S), direct P(E), P(E given S), and two
upper harm excesses. The conditional classification and
quantile losses ignore rows without S; they are not treated
as zero harm. The factored score uses P(S) times P(E given S).
The direct and factored scores share the same network and
training examples.

I reuse the 72 checked inner rankers and replay their 4,776
city-excluded training views; none of those rankers is
refitted. On calibration and held scenes, every arm uses
the exact existing upper-harm choice. The gain scores and
overall harm estimate are fixed. The comparisons are:

| Arm | Easy portion of the release score |
|---|---|
| Prior | Frozen selected-action veto |
| Support only | Prior score times predicted P(S) |
| Direct | Direct P(E) |
| Factored (registered primary) | P(S) times P(E given S) |
| Conditional upper tail | P(S) times the soft scores of both conditional upper margins |

All arms use the original exact standalone calibration:
both motion floors and all three 2% checks. Model scores
do not redefine the risk rule. The same 8/12 grid, city
roles, three seeds, 896 held development scenarios and
forecast menu are used. The result is not independent
confirmation.

## Which Part Improves?

There are 432 of 1,020 held two-slot seed views with an
edited-easy reference. Of those, 173 pass both easy checks.
Among these 173, 55 still fail the overall harm check and
43 more fail at least one motion floor. Only 75 satisfy
all three risk checks and both floors.

| Event prediction | Pooled AP | Pooled Brier | Positive rate |
|---|---:|---:|---:|
| Support | 0.691 | 0.1872 | 42.35% |
| Direct easy pass | 0.344 | 0.1322 | 16.96% |
| Factored easy pass | 0.345 | 0.1305 | 16.96% |
| Easy pass conditional on support | 0.458 | 0.2675 | 40.05% |

The paired city-balanced factored-minus-direct Brier
change is **-0.00323 [-0.00551, -0.00061]**. The conditional
classification head is still weak: its Brier score is
worse than the diagnostic constant-prevalence value
(0.2401), despite some discrimination.

| Upper coverage on the same supported choices | Prior | Conditional training | Paired change, pp [95% CI] |
|---|---:|---:|---:|
| Easy positive harm | 69.21% | 81.25% | +13.36 [+12.26, +14.60] |
| Signed easy harm | 65.05% | 75.69% | +12.72 [+8.45, +17.12] |

The coverage columns pool supported seed views. The paired
contrasts average seeds within each scenario and use 3,000
city bootstrap draws. Coverage is not the share that passes
the 2% rule. This is evidence for using conditional targets
on this fixed menu, not a guarantee about interventions.

## One Standalone Pass, No Matched Joint Gain

| Arm | Passing calibration prefixes / 1,020 | Held eligible / released |
|---|---:|---:|
| Prior | 3 | 0 / 3 |
| Support only | 3 | 0 / 3 |
| Direct | 3 | 0 / 2 |
| Factored | 2 | 1 / 3 |
| Conditional upper tail | 1 | No release |

The factored arm's successful standalone release is in
Austin, seed 43. It passes all original risk and accuracy
checks; easy net degradation is -2.24% (an improvement).
The same-budget fixed independent control also qualifies,
and beats this choice by 3.38 pp on the direct conservative
trajectory comparison and by 0.03088 native units on the
full-grid pair comparison. Thus the standalone pass is
**not** a matched joint success. Of the other two releases,
one has no edited-easy reference and one passes both easy
checks but has 43.55% overall harm and fails the dual gain
check. The city-balanced trajectory release result versus
CV is -0.00479 pp [-0.01437, 0]. I do not change deployment.

This changes the next experiment: preserve the separate easy
heads and test a *comparative* objective for both eligible
arms, including pair gain against the actual matched
independent path and conservative trajectory noninferiority.
Another standalone gate cannot resolve that gap.

## Reproduce

Requires the [city-excluded veto checkpoints](../av2_selected_veto_v1/README.md).

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_easy_gate.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_easy_gate.py score
.venv-pytorch/bin/python scripts/audit_m3w_av2_easy_release.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2*.py tests/test_m3w_interaction_frontier.py tests/test_m3w_slate_relative_gain.py
```

18 new arm64 CPU fits, four Torch threads, no loader workers.
Fit/freeze 35.19 s (655 MB peak RSS), held scoring 4.68 s,
case audit 5.14 s. Checkpoints every 40 steps retain
optimizer and sampler state. Completed receipt replay took
7.29 s. 59 focused tests passed. No new CREATE job,
manuscript edit, reserved confirmation access, Stage5C, or
SMC.

[Registration](registration.json) · [Freeze](freeze.json) ·
[Results](summary.json) · [Matched release audit](release_audit.json)
