# AV2: More Development Data Does Not Stabilize Release

The [small AV2 additive refit](../av2_learned_reference_v1/README.md)
rarely found interventions that both beat the motion floors and
passed risk. I expanded training to test whether sample size was
the main constraint, before spending more effort on joint ranking.

## Fixed Sampling and Comparison

I retained the original 128 scenarios and added 896 using the
first 64 scenario UUIDs in each first-hex prefix of the
[AV2 publisher-training archive](https://argoverse.github.io/user-guide/tasks/motion_forecasting.html).
The order was fixed from object listings before any added label
was inspected. The 1,024 files total 152,751,808 bytes. Neither
validation/test objects nor maps were downloaded. Publisher ETags
and local SHA-256 hashes bind the source. I found no identical
payloads or exact duplicate past rosters; that does not assert
that distinct scenarios in a city are independent recordings.

Both arms use the same observed roster, 8/12 grid, additive MLP,
320 optimizer steps, seeds 17/29/43, peak offsets and CV/OLS4
references. The small arm reuses verified checkpoints. The
large arm trains only on old and new scenarios from the same
four donor cities. Two-slot donor examples increase from 34-43
to 245-300 per fit. The easy/hard cuts are still the original
donor-only cuts so a change in the easy population cannot explain
a scale difference. Each arm calibrates on added scenarios from
the next city and scores on added scenarios from the held city.
The thresholds and forecasts are sealed before held scoring.

All 896 added UUIDs enter scoring across six held cities; 422
have complete futures for the whole roster. Of 84,864 requested
actor-steps, 79,389 are observed. The unit is AV2 native coordinates.
City folds are for development; these are not the reserved
confirmation set or an AV2 leaderboard comparison.

## Results on the Added Scenarios

| Forced policy | Known ADE | Available FDE | Conservative gain vs CV (pp) | Hard gain vs CV (%) | Risk passes / 1,932 edited views | Risk and both motion floors |
|---|---:|---:|---:|---:|---:|---:|
| CV | 0.24668 | 0.57458 | 0 | 0 | No edits | No edits |
| OLS4 | 0.27864 | 0.61665 | -22.03 | -10.91 | Unmatched | Unmatched |
| Fixed additive | 0.24510 | 0.56835 | -0.81 | +0.71 | 115 | 65 |
| Fixed joint | 0.24487 | 0.56783 | -0.76 | +0.71 | 117 | 63 |
| Small support | 0.24548 | 0.56957 | -1.06 | +0.48 | 118 | 61 |
| Large support | 0.24484 | 0.56789 | -0.78 | +0.90 | 122 | 64 |
| Large point-only | 0.24477 | 0.56799 | -0.69 | +0.87 | 118 | 66 |

Each matched arm makes 2,952 edits. Known ADE and FDE favor
some forced edits, but none beats CV on conservative utility.
In the small-to-large support comparison, conservative gain
improves by **+0.281 pp [-0.086, +0.673]** and hard gain by
+0.416 pp [-0.055, +0.957]. The direct known ADE difference is
-0.00064 [-0.00167, +0.00024]. Mean scenario p95 ADE falls from
0.68725 to 0.68486. Signed easy degradation exceeds 2% in 132
small and 118 large edited views. Undefined edited-easy ratios
increase from 1,633 to 1,644.

The direct full-grid pair effect is **-0.003133
[-0.003896, -0.002410]**; on 422 complete scenarios it is
+0.000915 [-0.000229, +0.002106]. This is a comparison between
additive forecasts, not evidence for a new joint policy. The
intervals use 3,000 paired city draws after averaging seeds
within each scenario.

## Release and Geographic Variation

| Diagnostic | Fixed additive | Small support | Large support |
|---|---:|---:|---:|
| Selected easy support, out of 1,020 two-slot seed views | 192 | 243 | 232 |
| Actual dual-gain and risk passes, same views | 50 | 46 | 49 |
| Both arms could qualify with a hindsight joint alternative | 41 | 37 | 40 |
| Nonempty additive-only calibration thresholds passed | Not tried | 8/1,020 | 3/1,020 |
| Held edited views under the selected threshold | Not tried | 3 | 0 |
| Held edited views meeting dual gain and risk | Not tried | 2 | 0 |

There are 624 two-slot views where a legal assignment has
edited-easy support. The learned forecasts still use less than
half of these. Expansion improves the two-slot qualification
count in Austin (14 to 25), but decreases it in Miami (18 to 3).
The calibration rows are scenario views, not independent
recording-level certificates. One of the small arm's three held
interventions exceeds the signed easy limit; the large arm
returns CV everywhere. Neither is a robust deployable result.
I did not evaluate a new joint release. No risk rule changed.

The additional data improve average forecast accuracy modestly
but do not stabilize risk support or justify another joint
ranker on the same menu. The next priority is a donor-trained
public 8/12 forecaster as a matched candidate producer, using
this larger development roster to test whether forecast quality
and support can improve together before reserving a policy for
external confirmation.

## Reproduce

Requires the previous [expanded AV2 and additive fits](../av2_learned_reference_v1/README.md).
On the Mac:

```bash
export SSL_CERT_FILE=/etc/ssl/cert.pem
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/acquire_m3w_av2_development.py
.venv-pytorch/bin/python scripts/run_m3w_av2_data_scale.py inputs
.venv-pytorch/bin/python scripts/run_m3w_av2_data_scale.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_data_scale.py score
```

Acquisition took 75.41 s, input generation 37.70 s, 18 native arm64
fits 24.44 s and the verified scoring pass 25.91 s, with maximum
RSS below 885 MB. Four Torch threads, no loader workers; download
receipts and 40-update model checkpoints support resume. Each
fit/scoring phase records PID and progress. The first scoring
pass completed per-scenario metrics but could not serialize a
NumPy count. The fixed readout verified the archived source hash,
rescored the same frozen predictions and wrote the summary.
No model was retrained for that repair.

Thirty-one focused tests passed. No CREATE job, reserved labels,
Stage5C, SMC, manuscript or deployment change. Raw data, fit
weights and full per-scenario manifests remain local. The report
links to a snapshot of the [executed protocol](executed_protocol.json)
and [reported metrics](reported_metrics.json); the commands write
their own local registration and summary.
