# Does donor noise training help with camera histories?

I trained the same spline six times on seven OpenDD development
locations: clean and perturbed inputs at each of three paired
seeds. Each run used 800 updates. I froze the predictions before
scoring the previously exposed WEPDTOF Convenience Store, GMU and
CMU camera prefixes. No target camera labels were used to train or
select the models. The synthetic perturbation models correlated
position jitter, not missed detections or identity changes.

## Direct forecasts

| Development camera | Online / stable match / full future | OLS4 known ADE (px) | Median slope known ADE (px) | Clean spline known ADE (px) | Noise spline known ADE (px) | Noise vs OLS4 known-step gain (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WEPDTOF Convenience Store | 58 / 57 / 57 | 14.61 | 13.95 | 17.44 to 18.60 | 13.48 to 13.73 | +3.33 to +4.71 |
| GMU | 32 / 8 / 5 | 58.76 | 56.81 | 87.80 to 91.54 | 72.29 to 74.48 | -19.41 to -15.88 |
| CMU | 366 / 8 / 7 | 31.63 | 31.66 | 44.36 to 44.86 | 32.87 to 33.11 | -2.09 to -1.66 |

ADE is the mean of per-track known-step errors. Paired gain
instead weights queries equally and uses known CV error mass as
the denominator. The seed ranges are not confidence intervals.
The WEPDTOF conservative gain over OLS4 is +2.49 to +4.36 pp
and over median slope is -0.88 to +0.97 pp. The direct noise
spline passes 0/3 original risk checks: easy positive-harm
upper bounds are 29.67% to 33.84% against a 2% limit.

The noise spline has lower known-step error than the clean spline
at all three cameras. It does not beat OLS4 on known-step ADE
at GMU or CMU. Only eight stable histories are available on
each ACME camera; many unmatched online histories have no
released outcome. Keeping them makes conservative bounds
strongly negative. In particular, CMU's 358 wholly unavailable
histories and small known CV error mass make the lower bound
large in magnitude; that bound is not an observed error increase.
CMU has lower available-endpoint FDE for the noise spline than
for OLS4; GMU does not. The full metrics, including the CV
comparator and unavailable rows, are in [readout.json](readout.json).
No new easy cutoff was fitted on either ACME camera.

## Can the new bank help if the policy can take full-size actions?

On the WEPDTOF pilot, preserve the original eligibility and
per-query edit caps (23 slots per seed). Compare OLS4 and median
slope alone with the same bank plus both splines. The original
clipped bank is excluded by the original risk-constrained
upper bound for both menus in 3/3 seed views. The full paths
yield verified hindsight witnesses in 3/3 views for each menu.

| Seed | Analytic only, full-path lower gain vs OLS4 (pp) | With both splines (pp) | Difference (pp) |
| --- | ---: | ---: | ---: |
| 17 | +8.73 | +15.34 | +6.61 |
| 29 | +8.15 | +14.50 | +6.35 |
| 43 | +8.73 | +13.44 | +4.72 |

Each witness is re-scored against CV and OLS4 under the
original all/easy/signed 2% checks, including positive reference
support. This isolates an action-space restriction; it does not
show that greater movement is safe in use. The neural difference
is between two menus and does not isolate which spline contributes.
Future outcomes select the witnesses. No past-only policy was
trained for the full-path bank, and the current fallback remains CV.
The next experiment is to learn gains and harms for full-size
actions on out-of-fold donors, then compare past-only independent
and joint choice at the same realized counts and displacement.
Keep the analytic-only bank as a control.

## Run record

Fresh native arm64 Torch, four compute threads, no DataLoader
workers. Six fits, 4,800 updates total, 100-step checkpoints and
heartbeats. Training and freeze took about 16 seconds; scoring
took about four seconds. Re-loading the completed checkpoints
reproduced the forecast arrays and score receipt exactly. The
capacity run also replayed exactly. Ten focused tests passed.
No CREATE submission, no new training labels from the camera
cohorts, and no scoring of the 13 WEPDTOF external groups,
Miraikan, or sealed DroneCrowd confirmation. No metric/seconds,
Stage5C, SMC, deployment or submission claim.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_camera_noise_control.py freeze
.venv-pytorch/bin/python scripts/run_m3w_camera_noise_control.py score
.venv-pytorch/bin/python scripts/audit_m3w_camera_noise_capacity.py
```

[Protocol](protocol.md) | [Freeze](freeze.json) |
[Capacity](development_capacity.json) |
[Working manuscript v162](../evidence_manuscript_v162/README.md)

Related work: [Yu and Zhou](https://arxiv.org/abs/2108.08259)
study re-tracking and temporal consistency on raw videos;
[NATRA](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_NATRA_Noise-Agnostic_Framework_for_Trajectory_Prediction_with_Noisy_Observations_ICCV_2025_paper.pdf)
learns to denoise observed histories; [RealTraj](https://research.nvidia.com/labs/twn/publication/icra_2026_realtraj/)
forecasts directly from detections. None is reproduced by this
simple donor perturbation. The question here is whether a
candidate can be used as a reference-relative action, not whether
adding noise to training data is novel.
