# Weighted velocity on the existing OpenDD held-location roster

**Result: better label coverage did not make this model beat the stronger
reference.** I refit the weighted-velocity architecture from scratch on
the existing exposed OpenDD pedestrian development cohorts. This is a
development control, not zero-shot transfer or independent confirmation.

The original cache receipts and role registration were verified before
use. Seven locations, 202 recordings and 10,475 windows are retained,
with one location excluded from each fit. This uses the existing
8-observation/12-prediction OpenDD protocol at source stride 12, not the
same sampling interval as MovingDroneCrowd++. No new absolute time or
physical unit comparison is inferred.

OpenDD supplies 121,211 of 125,700 requested labels and 9,823 complete
windows; 53 are wholly unlabeled. Its normalized reference-disagreement
floor is 0.282 pp, compared with 11.551 pp on the original moving-drone
roster. These are within-corpus relative scores, not comparable
absolute errors. Different motion, sampling, and source support prevent
a causal claim that annotation coverage alone explains the performance.

| Direct contrast on the OpenDD roster | Conservative gain (pp) | Paired 95% location CI |
|---|---:|---:|
| ADE-trained weighted velocity vs CV | -4.568 | [-6.939, -1.837] |
| Dual-loss weighted velocity vs CV | -4.794 | [-7.255, -2.039] |
| Dual-loss weighted velocity vs OLS4 | +4.188 | [+2.933, +5.278] |
| Dual loss vs ADE with matched architecture | -0.228 | [-0.352, -0.117] |
| Dual loss vs fixed robust fit | +6.713 | [+3.760, +9.641] |

The strongest of these two references here is CV, not OLS4.
All full-path arms pass 0/606 original recording-seed risk views.
The dual arm has 30 undefined easy-risk views and worst defined signed
easy degradation of 392.20%; it is not deployable. The 606 views are
repeats, not independent units. Inference and error comparison are
paired by location, using 3,000 draws after averaging three actual
seeds. The data are strongly unbalanced by location (12 to 5,349
windows), so this is not a large external confirmation set.

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_weighted_velocity.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_weighted_velocity.py --replay
```

Native arm64, four Torch threads, zero workers. Forty-two fresh fits
(seven locations, two matched objectives, three seeds, 600 updates per
fit) took 23.57 s. A 1.87 s checkpoint-only replay reproduced the same
analysis. No role, calibration, policy, deployment, Stage5C or SMC change.

[Analysis](analysis.json) | [Raw support audit](../moving_future_support_v1/README.md) |
[Working supplement v49](../evidence_manuscript_v49/README.md).
