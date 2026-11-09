# Learning which full trajectories to use

I tested whether learning complete edit plans helps where estimating
gain and harm separately did not. It produces interventions, but
does not establish a reliable advantage from joint selection.

The teacher finds a plan for each training recording using future
labels and the original CV-referenced 2% rule. It verifies 157
positive analytic plans on 202 recordings. Thirty-four records
have a certificate of infeasibility or nonpositive utility; eleven
lack support and are not treated as negative examples. The expanded
bank gives 533 verified plans across 606 training recording-seed
views. These training plans are not test results.

The student ranks zero, one or two edits using the observed
history and candidate paths. Additive and pair-coupled scores
share the encoder and examples. The first experiment uses only
analytic candidate forecasts and excludes the scored location
from fitting. The camera experiment uses the existing held-site
neural forecasts for donor supervision and all-donor frozen
forecasters at inference. The latter scores the single exposed
Convenience Store camera, not the 13 external groups.

## OpenDD: seven held development locations

| Policy | Recording-seed views | Edits | Original risk passes | Risk and both utility passes |
| --- | ---: | ---: | ---: | ---: |
| CV fallback | 606 | 0 | 0 | 0 |
| OLS4 | 606 | 29,685 | 0 | 0 |
| Joint | 606 | 3,690 | 41 | 34 |
| Independent | 606 | 2,159 | 40 | 36 |
| Same-count, same-peak independent | 606 | 3,690 | 39 | 32 |
| Same-count, same-peak common family | 595 | 3,321 | 39 | 32 |

The two matched controls fill the exact peak slots selected by
the joint policy. An alternative forecast direction may be
shortened but not enlarged. The common-family control cannot fill
all slots in eleven views. The independent policy without matching
uses fewer edits, so its counts cannot isolate the joint effect.

Joint minus matched independent lower utility is **-0.022 pp**
(95% location bootstrap **[-0.070, +0.016]**; six of seven
estimable sites). The matched common-family result is -0.021 pp
[-0.063, +0.012], also six sites. A site with no edit does not
supply a matched utility estimate. These intervals resample
locations after averaging recordings and seeds; they are
descriptive because the source folds overlap in training.

Across the 606 paired pass outcomes, 3 views pass only with joint
and 1 only with matched independent. All of the net difference
comes from one location. The seven-location mean difference is
+0.075 pp [0.000, +0.225]. The joint arm abstains in 167 views,
has undefined easy support in 117 active views, fails measured
risk in 281, and passes risk but not utility in 7.
Median paired tail and relative-path error differences are zero
at each location. See [diagnostics](diagnostics.json) for native
units and denominators. Relative-path error is not physical safety.

## Exposed camera: more edits, still no deployment pass

| Seed | Joint edits | Conservative gain vs CV (pp) | vs OLS4 (pp) | Easy positive-harm upper bound |
| --- | ---: | ---: | ---: | ---: |
| 17 | 21 | +9.47 | -16.91 | 11.73% |
| 29 | 20 | +12.39 | -13.98 | 38.79% |
| 43 | 22 | +8.66 | -17.72 | 17.40% |

No edited track has an unavailable future. Easy harm is observed,
not just a consequence of completion bounds. The joint rule
beats matched independent in two seed views and loses in one;
it loses to the matched common family in all three. These are
three fits evaluated on one camera, not three validation sites.
No arm qualifies for deployment. CV fallback remains unchanged.

The next priority is a development camera cohort with enough
stable matches for source-separated gain and harm fitting.
Use it to test a source-independent risk gate against an
analytic selector before training a more expressive pair model.
The reserved WEPDTOF outcomes and sealed DroneCrowd confirmation
are not training or calibration resources.

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_slate_imitation.py freeze
.venv-pytorch/bin/python scripts/run_m3w_slate_imitation.py score
.venv-pytorch/bin/python scripts/audit_m3w_slate_imitation.py
```

Native arm64 Torch, four compute threads and no workers.
24 fits, 19,200 updates; freezing took 47 seconds and scoring
about three seconds. Each head has 100-step checkpoints.
Twelve focused tests passed; completed checkpoints reproduce
the frozen path arrays and scores. The roster check verifies
the existing windows are disjoint through the twentieth
sampled step, with no new data split. No CREATE job or
independent scoring. No Stage5C, SMC, metric or seconds claim.

[Protocol](protocol.md) | [Freeze](freeze.json) |
[Readout](readout.json) | [Working manuscript v164](../evidence_manuscript_v164/README.md)

This is imitation of verified plans, not a reproduction of
[decision-focused relaxation](https://ojs.aaai.org/index.php/AAAI/article/view/3982)
or [SPO+](https://pubsonline.informs.org/doi/10.1287/mnsc.2020.3922).
