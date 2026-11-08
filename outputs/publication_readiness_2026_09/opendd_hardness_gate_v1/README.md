# Does Predicting CV Difficulty Identify Useful Neural Corrections?

The [component audit](../opendd_component_guard_v1/README.md) finds
hard-case gains and easy-case losses. Here I separate difficulty
prediction from neural candidate quality. A small head predicts
CV-easy, CV-hard and other from eight observed positions, causal
velocities, and CV/OLS4 errors on short suffixes of that same observed
window. Its labels are defined by CV error, never by the frozen neural
forecast's win/loss. Each evaluation site is excluded from training
and cut estimation. Within training, each site's own labels are
excluded from estimating its cuts.

## Prediction And Matched Decisions

Hard AUROC is 0.650 versus 0.611 for normalized past CV/OLS4
disagreement; AUPRC is 0.389 versus 0.399 (site-macro positive rate
0.252). Easy AUROC is 0.763. These are descriptive seven-site
macro-averages after averaging seeds. They are not calibration or
policy performance.

The frozen decision rule switches to ego-only EqMotion when hard
probability is at least 0.50 and easy probability below 0.25. The
simple motion control uses the **same count in each query** but picks
by CV/OLS4 disagreement. Both use the same raw trajectory bank, not
matched radii in this first comparison. A separate matched-radius
comparison permits changing only the path shape at the existing slots.

| Raw rule | Edits | Hard among selected | Easy among selected | Undefined views | Joint passes / 606 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full neural forecast | 30,075 | 7,155 | 8,319 | 30 | 6 |
| Learned difficulty gate | 3,323 | 1,640 | 454 | 437 | 6 |
| Same-count motion control | 3,323 | 1,621 | 472 | 438 | 5 |

The learned rule's lower gain is **-0.277 pp [-0.905, +0.227]**
versus CV, and **+0.070 pp [-0.275, +0.476]** directly versus the
same-count control (3,000 site bootstrap draws after seed averaging).
Selection increases hard concentration versus the full neural policy,
but only slightly versus the count-matched causal control. The
baseline and recording gate remain unchanged; abstention does not
create easy-reference support.

At the original 1,457 slots, the gate changes 185 donor-prefix shapes
to neural shapes at the same radii. Joint passes fall **90 to 84**.
Direct effect is **-0.0185 pp [-0.0734, +0.0126]**. No learned rule
advances. Predicting where CV may struggle is not the same as predicting
where a particular alternative will reduce error and meet the risk rule.
The next diagnostic should separate candidate advantage from difficulty
on entirely out-of-source forecasts, rather than keep tuning this
hardness threshold.

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_opendd_hardness_gate.py
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_opendd_hardness_gate.py --replay
```

21 native arm64 CPU Torch heads, 8,400 fresh updates, four compute
threads, zero data workers. Checkpoints include optimizer and sampler
state and heartbeat every 100 steps. The first run took 7.2 seconds;
zero-update replay took 3.6 seconds and agrees with the frozen results.
19 focused tests passed across the component, gate, original forecaster,
and confirmation-role code. Data, learned weights and raw predictions
remain local. All 606 views repeat 202 exposed recordings at three seeds.
No deployment, external calibration, confirmation, Stage5C or SMC.

[Registration](registration.json) | [Results](analysis.json) |
[Checkpoint receipts](head_freeze.json)
