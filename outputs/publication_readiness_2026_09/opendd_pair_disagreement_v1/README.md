# OpenDD: where the joint selector loses its advantage

I replayed the frozen two-edit decisions to separate three explanations:
the joint policy might pick worse trajectories, pick recipients with
incomplete future labels, or lose the easy reference support needed to
pass the release rule. This uses the original seven development locations,
493 unique recordings, three seeds and six calibration directions per
location/seed. The 8,874 views per comparison are repeated measurements.
All original per-direction trajectory and risk counts reconcile.

## Matched trajectory results

| Treatment minus control | Observed gain (pp; 95% site CI) | Missing-step debit (pp) | Conservative lower (pp) | Missing-step upper (pp) |
| --- | ---: | ---: | ---: | ---: |
| Learned joint vs learned additive | +0.000249 [-0.000452, +0.000958] | 0.005580 | -0.005331 | +0.005829 |
| Learned joint vs same-family | -0.000435 [-0.000827, -0.000139] | 0.000563 | -0.000998 | +0.000129 |
| Correct spacing vs motion only | +0.000110 [-0.000031, +0.000248] | 0.001986 | -0.001876 | +0.002095 |

For each missing step, the reverse triangle inequality bounds the
unknown paired gain by plus or minus the distance between the two
predictions. Summing those distances gives the debit. Lower and upper
endpoints use the same observed CV error mass as the original
comparison; neither endpoint estimates the missing ground truth.
Intervals use 3,000 paired bootstrap draws over the seven location
means, not over individual windows. Exact estimates, component
intervals, site values and gate transitions are in [diagnostic.json](diagnostic.json).

Compared with learned additive, joint changed the recipients in 6,290
views and changed only the paths in another 89. Of the 6,379 differing
views, 3,791 had a missing label on at least one changed prediction step.
The negative conservative contrast comes from charging these missing
steps, not from a demonstrated increase in observed error. But when
all changed steps were observed, the 2,588 views contributed
-0.000190 pp to the overall contrast, using the same denominator.
This diagnostic subset is defined after labels and is not an inference
rule. Removing the family restriction, unlike adding true spacing,
also lost observed gain with a site interval below zero.

## The recording-level check is a separate bottleneck

Compared with additive, learned joint lost 1,087 views that previously
beat CV and OLS4 and met the original risk checks, and gained 1,064,
for a net loss of 23. The largest single loss was 708 previously
qualifying views whose joint edits had no easy reference denominator.
The reverse transition recovered 500. No denominator is not a measured
harm event, and it does not count as a pass.

The practical implication is to learn when a proposed edit will have
enough easy reference support, alongside its gain. Simply adding
spacing or expanding the set of legal combinations has not solved
that problem. A support-aware ranking would still need to show positive
paired utility under the same budget, not just fewer unscorable edits.
This is a candidate next experiment, not a deployed method.

## Paper interpretation

The result supports a narrow claim: changing the decision rule changes
which evidence is available, even with the forecaster and edit budget
fixed. It does not support improved joint prediction, a population risk
guarantee, or external transfer. In contrast to [Wilder et al.](https://ojs.aaai.org/index.php/AAAI/article/view/3982),
the frozen heads in this study predict pair outcomes before exact
enumeration; they do not backpropagate through an optimizer. The
recording-level pass rule is also not the population risk procedure
studied in [Learn then Test](https://people.eecs.berkeley.edu/~angelopoulos/publications/downloads/ltt.pdf).

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/diagnose_m3w_opendd_pair_disagreement.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_opendd_pair_disagreement.py tests/test_m3w_vehicle_pair_learning.py tests/test_m3w_vehicle_pair_geometry.py tests/test_m3w_vehicle_pair_support.py tests/test_m3w_songdo_confirmation.py
```

`fresh_run`: paired attribution and original risk replay on real
development labels. `cached_verified`: source cohorts and frozen
choice packets. Eighteen tests passed. No new training or thresholds,
external label access, HPC submission, manuscript edit or deployment.
