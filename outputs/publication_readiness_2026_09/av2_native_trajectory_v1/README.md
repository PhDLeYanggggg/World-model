# Argoverse 2: Does Joint Selection Improve With Denser Labels?

The Okutama comparison left two explanations open: incomplete future
annotations could hide a benefit, or the frozen selector could fail to
find useful joint changes. I tested the same decision layer on a denser
map-frame source. This is an annotated-trajectory development experiment,
not an aerial tracking experiment or an AV2 leaderboard entry.

## Data And Controls

I sampled 128 scenario Parquet files (19,455,729 bytes) from the
[Argoverse 2 Motion Forecasting](https://www.argoverse.org/av2.html)
publisher training split. The first eight UUIDs in each first-hex stratum
were fixed from the public object listing before download. No map,
validation or test objects were downloaded. There was one query per
scenario. Of the 1,014 actors admitted using timesteps 42–49, 896 have all
twelve future annotations. Overall coverage is 11,358/12,168 steps
(93.34%). The main roster is mixed, not pedestrian-only. The 126
publisher-labelled pedestrians and 33 scenarios with co-observed
pedestrian pairs are separate scoring slices, not inference filters.

I computed motion from consecutive past positions. Publisher velocities,
scored-track categories, maps and future rows did not enter inference.
The OLS4 hindcast nominated at most two edit slots. All edited methods
had identical counts and peak offsets within each scenario. The
scene-uniform control applies one common trajectory family to the
anchor's recipients; additive and joint methods use the same legal
menu. They reuse three existing OpenDD seed ensembles, each composed
of seven verified location folds. Forecasts were hashed before
scoring, and AV2 did not enter training or threshold selection.

## Results

| Forecast | Complete ADE | Available FDE | Full-grid ADE gain vs CV (pp) | Risk passes among edited views |
|---|---:|---:|---:|---:|
| Last-step constant velocity | 0.2627 | 0.5773 | 0.00 | Not an edited policy |
| OLS4 | 0.3116 | 0.6502 | −29.30 | Not matched |
| Damped velocity | 0.7236 | 1.7224 | −226.03 | Not matched |
| Tapered turn | 0.3015 | 0.6326 | −26.95 | Not matched |
| Past-hindcast anchor | 0.2725 | 0.5904 | −3.69 | 9/285 |
| Scene-uniform shape | 0.2724 | 0.5903 | −3.79 | 9/285 |
| Additive | 0.2712 | 0.5880 | −3.47 | 2/285 |
| Risk joint | 0.2714 | 0.5886 | −3.57 | 3/285 |
| Relative joint | 0.2714 | 0.5886 | −3.57 | 3/285 |

ADE/FDE are in AV2 native units. Gain averages scenario-level
missingness-debited gains normalized by observed CV error; it is not the
ratio of the ADE entries. Seeds are averaged within scenario and
scenarios within city, with equal city weight. Relative joint loses
3.57 pp to CV (3,000-draw paired city bootstrap interval −5.99 to −1.18).
The known hard-subset point estimate is −1.44%. CV is the strongest of
the four tested motion baselines. Each intervention arm makes 459 edits
across three seeds (15.09% of 3,042 actor-views). For relative joint, 33
views exceed the 2% signed easy threshold and 236 edited views lack
an edited-easy denominator. These are not passes. Easy references are
other-city complete-CV error quartiles used for scoring only; these
short scenario views do not certify recording-level risk.

| Relative joint vs matched additive | Full-grid pair gain (native units) | 95% city bootstrap interval |
|---|---:|---:|
| All 128 scenarios | −0.000541 | [−0.000846, −0.000236] |
| 59 fully labelled scenarios | −0.000008 | [−0.000146, +0.000100] |
| 27 fully labelled two-slot scenarios | +0.000001 | [−0.000243, +0.000208] |

All six city means are negative on the full-grid pair contrast.
On co-observed steps alone, the gain is −0.000232
[−0.000516, +0.000018]. Missing steps affect the lower bound, but
even with complete labels there is no reliable joint gain.
The pedestrian slice does not rescue it (full-grid pair gain
−0.001430 [−0.002593, −0.000446]). There are only six city clusters;
these intervals describe the bounded development sample.

In the exact hindsight scan, 73/81 complete two-slot seed views
admit a better pair assignment before accuracy and risk checks.
Only 15/81 (five unique scenarios) can improve pair error while
beating CV and OLS4, preserving additive ADE and passing all three
original 2% checks. The learned selector finds zero such wins.
Both the menu and the ranking limit performance.

## Paper Implication

This is a useful negative control: matching edit count and peak offset
separates changes in the chosen combination from changes in intervention
intensity. It is not evidence of joint-policy generalization. Joint
forecasting is established work ([Scene Transformer](https://arxiv.org/abs/2106.08417),
[QCNeXt](https://arxiv.org/abs/2306.10508)); here the question is which
parts of a baseline forecast to replace when only limited edits are
allowed. Published AV2 scores are not matched comparators for this
train-source sample, roster, input modalities or forecast grid.

The next test should change the causal candidate paths. First measure
whether a new producer creates more strict hindsight wins on
development sources under the same constraints; retrain the decision
layer only if there is more feasible headroom. Deployment and
independent-confirmation status are unchanged. Stage5C and SMC remain off.

## Reproduce

Use the native arm64 environment with PyArrow and the frozen OpenDD
checkpoints. On macOS the system CA bundle verified the public endpoint.
No CREATE job was needed.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export SSL_CERT_FILE=/etc/ssl/cert.pem
.venv-pytorch/bin/python scripts/run_m3w_av2_native_trajectory.py acquire
.venv-pytorch/bin/python scripts/run_m3w_av2_native_trajectory.py freeze
.venv-pytorch/bin/python scripts/run_m3w_av2_native_trajectory.py score
.venv-pytorch/bin/python scripts/report_m3w_av2_native_trajectory.py
```

Each completed download and forecast is checkpointed; the local
heartbeat records phase, PID and completion. Freeze and scoring took
3.95 s and 3.30 s respectively on this machine, each under 481 MB
maximum RSS. The focused test suite passed 31 checks.
[Registration](registration.json) and [metrics](summary.json) are tracked.
Raw data, forecasts, detailed scoring rows and weights stay local.
