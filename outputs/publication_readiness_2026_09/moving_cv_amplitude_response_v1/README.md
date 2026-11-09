# Learning how far to edit

I tested whether a model could learn the magnitude of a trajectory
edit after the previous study ruled out gains at the inherited peak
budget in most development views. This changes the model, not the
original recording-level 2% rule.

**Design.** Use eight observed and twelve forecast steps in local
pixels. Two models predict gain and harm at five fractions
(.05, .15, .35, .65, 1) of each causal candidate ray. The direct head
fits free knots; the shape head encodes concave lower gain and convex
harm and signed easy change. This follows from the Euclidean error
along a fixed ray and the linear debit for missing steps; it does not
require a general input-convex network. General shape constraints are
related to [Amos et al.](https://proceedings.mlr.press/v70/amos17b.html)
and [Gupta et al.](https://proceedings.mlr.press/v119/gupta20b.html).
Training and calibration use analytic candidates; held development
also uses verified source-excluded neural proposals. Neither future
positions nor future validity masks enter inference. The head predicts
easy reference mass separately from harm; ease probability is auxiliary.
Both models share sampling, capacity and training budget: 18 arm64
PyTorch fits, three source-role rotations, three seeds, 25,200 updates.

**Held development.** 24 exposed recordings, nine source groups,
72 recording/seed views. Values are paired percentage points (pp);
positive favors the first policy. Intervals resample nine sources,
after averaging seeds (3,000 draws).

| Contrast | Observed-only pp | Completion-lower pp (95% CI) |
|---|---:|---:|
| Shape vs direct independent ranking, same per-query edit counts | -0.081 | -2.341 [-3.418, -1.268] |
| Shape portfolio vs independent shape selection, same per-query edit counts | -0.620 | -2.875 [-6.466, -0.816] |
| Adaptive vs uniform amplitude, same recipients, rays and per-query sum of peaks | +0.071 | -0.380 [-1.006, -0.011] |
| Shape portfolio vs constant velocity | +0.420 | -1.074 [-3.064, +0.078] |
| Shape portfolio vs OLS4 | -0.181 | -55.268 [-96.563, -23.048] |
| Twice the frozen edit vs constant velocity | +1.024 | -0.747 [-1.666, +0.073] |

The completion-lower score is a conservative paired bound, not a
measurement of unobserved positions. The equal-mass control holds
48,915 edits and a total of 367,255 local pixels of per-row peak
displacement (rounded); matching edit counts alone does not match
displacement. Doubling the frozen edits raises worst known signed
easy change from 61.90% to 128.41%. The shape portfolio's worst known
signed easy change is 6.31%; only 1/72 views passes the original risk
check. The observed-only diagnostic would pass 15/72; 14 of those fail
when scored with the original missingness treatment. Calibration
qualifies 0/9 source-role/seed cells. Undefined support is not a pass.

**Why it failed.** The shape head reduces normalized dual-gain MAE
by 0.0090 [0.0018, 0.0161] and signed easy MAE by 0.0137
[0.0102, 0.0171] on sampled held analytic rays. The average error
reduction is real on this diagnostic, but the end-to-end policy
does not improve at matched count or matched mass. The observed
benefit of adaptive amplitudes does not survive uncertainty on
unlabeled steps. Some edits also harm observed easy cases, so
missingness alone does not explain the failure. The next test is
whether training for tail harm of the edits actually selected,
rather than for average ray fit, improves the same matched controls.

**Provenance and replay.** Training, development scoring, and the
response audit are `fresh_run`. The frozen donor and anchor artifacts
are `cached_verified`. The checkpoint replay reproduced the frozen
readout without optimizer updates. Anonymous clean-extraction replay
reproduces the scalar risk and utility tables, not trajectory data,
geometric contrasts or model training.

```sh
.venv-pytorch/bin/python scripts/train_m3w_cv_amplitude_response.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_amplitude_response.py
.venv-pytorch/bin/python scripts/build_m3w_cv_amplitude_response_replay.py
```

To fit or resume, omit `--replay`. Checkpoints, lock, and heartbeat
are in the ignored local experiment directory. This is an 8/12
local-pixel study, not metric, seconds-level, true 3D or foundation
evidence. No deployment change, Stage5C or SMC. Independent
confirmation is `not_run`: no qualifying policy. The agreed
DroneCrowd protocol remains 15 source groups for calibration and
30 distinct source groups sealed for independent confirmation.

[Protocol](protocol.md) | [Readout](readout.json) |
[Response diagnostic](response_audit.json) | [Supplement v139](../evidence_manuscript_v139/README.md).
