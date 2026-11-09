# Does Past Camera Motion Explain the Okutama Transfer Gap?

The [label-history experiment](../okutama_label_histories_v1/README.md)
left camera motion in the trajectories. I tested that nuisance directly
on the first stride-ten, nongenerated-label window with at least two
complete past histories in each of the 33 publisher-training clips.
The window and roster depend only on observations, not future support.
Both camera views of the same activity stay together, giving 17
scenario groups at one facility.

I read exactly 264 past JPEGs. The ZIP directory matched the previous
receipt, and each JPEG passed the member CRC, local SHA-256 and raster
shape checks. The download was 36,422,299 bytes including the ZIP index,
not the full archive. I did not retrieve future images or test files.

## Comparison

On each observed frame pair, background optical flow excludes the
visible nongenerated person boxes, then fits a similarity transform.
If all seven fits are accepted, I align the eight past positions to
the final observed image plane, then run the same CV or OLS4 forecast.
The align-only version holds the camera fixed. The second version
also extrapolates the image-plane transform, using the median center
shift, rotation and log-scale from the last three observed pairs, with
0.9 decay per future step. Both versions return forecasts in the
original image coordinates. If one fit fails, the arm returns the
matching raw forecast exactly. This is an image-plane ablation, not
3D ego-pose or physical calibration. For related learned camera-motion
forecasting, see [Neumann and Vedaldi](https://openaccess.thecvf.com/content/CVPR2021/papers/Neumann_Pedestrian_and_Ego-Vehicle_Trajectory_Prediction_From_Monocular_Camera_CVPR_2021_paper.pdf).

168 of 231 pairs pass the background fit; 19 of 33 windows activate
compensation. The fixed roster has 198 histories, 89 complete futures,
15 windows with at least two complete futures, and 12 fully labelled
rosters. I score partial futures rather than select complete cases.
Bootstrap intervals use 3,000 paired scenario draws, averaging cameras
within scenario. They describe this facility only.

| Policy | Observed ADE, px | Available-endpoint FDE, px | Conservative individual gain vs matching raw, pp [95% CI] |
|---|---:|---:|---:|
| Raw CV | 185.05 | 467.46 | reference |
| CV, align past | 204.66 | 457.52 | -53.99 [-127.90, -2.46] |
| CV, align and reproject | 176.61 | 406.58 | -7.53 [-22.76, +3.86] |
| Raw OLS4 | 189.59 | 455.38 | reference |
| OLS4, align past | 211.31 | 453.44 | -59.27 [-141.29, -1.70] |
| OLS4, align and reproject | 181.54 | 396.87 | -7.12 [-24.51, +5.62] |

The FDE improvement is paired: +60.88 px [17.00, 112.17] for CV and
+58.51 px [18.86, 105.04] for OLS4. However, CV's +5.71 pp observed
gain minus a 13.23 pp missing-step debit leaves -7.53 pp; OLS4's
+6.70 pp minus 13.82 pp leaves -7.12 pp. The debit bounds unobserved
error changes; it does not measure observed harm. For CV, scenario
2.4 has +38.38 pp observed gain but -80.84 pp conservative gain.
On the 19 fit-supported windows, the conservative individual
intervals still span zero. Conservative pair gains also span zero
for both reprojected arms: CV -0.47 px [-2.43, +1.94], OLS4
+0.36 px [-1.48, +3.03].

**Decision:** camera extrapolation helps the visible endpoint in this
diagnostic, but does not explain away the frozen transfer failure or
establish a whole-grid joint gain. Keep it as an ablation rather than
a deployment change. A stronger next experiment needs both usable
camera motion and denser independent future support. No training, HPC
job, confirmation access, Stage5C or SMC.

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_okutama_past_camera.py acquire
.venv-pytorch/bin/python scripts/run_m3w_okutama_past_camera.py freeze
.venv-pytorch/bin/python scripts/run_m3w_okutama_past_camera.py score
.venv-pytorch/bin/python scripts/report_m3w_okutama_camera_debit.py
.venv-pytorch/bin/python scripts/plot_m3w_okutama_past_camera.py
```

Acquisition resumes from verified per-image receipts and writes a
PID heartbeat. Freeze and score can be replayed offline. The paired
vector figure is written to the private experiment directory. No
images or prediction arrays are committed.

[Registration](registration.json) | [Past image receipt](image_freeze.json) |
[Readout](readout.json) | [Missing-label attribution](debit_attribution.json)
