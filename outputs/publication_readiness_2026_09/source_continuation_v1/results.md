# Full Source-Training Continuation Results

## Material Passport

Fresh continuation of three verified parent models into six paired schedule branches.
48000 new updates; shared parents contain6000 unique inherited updates. No held-source/main scoring.
All15430 training rows retained. Mean and range of three seeds, not generalization confidence intervals.

| Schedule | Step | Mean gain (%) [seed range] | Moving gain (%) | Hard gain (%) | Easy absolute pixel harm |
| --- | ---: | --- | ---: | ---: | ---: |
| constant | 2000 | -1.570015 [-2.054110, -1.189468] | -0.306941 | -0.035175 | 0.02565618 |
| constant | 4000 | -1.231036 [-1.284508, -1.196214] | -0.142671 | +0.015683 | 0.02210739 |
| constant | 6000 | -1.494004 [-1.590169, -1.308893] | -0.001717 | +0.108549 | 0.03031207 |
| constant | 10000 | -1.031000 [-1.213010, -0.842413] | +1.244599 | +0.874670 | 0.04622308 |
| cosine | 2000 | -1.570015 [-2.054110, -1.189468] | -0.306941 | -0.035175 | 0.02565618 |
| cosine | 4000 | -1.155071 [-1.461442, -0.927605] | -0.142599 | +0.011382 | 0.02056583 |
| cosine | 6000 | -0.768717 [-0.942741, -0.546379] | +0.141998 | +0.115197 | 0.01849890 |
| cosine | 10000 | +0.253283 [+0.203854, +0.325909] | +0.876220 | +0.415533 | 0.01265339 |

## Verification and Limits

All24 milestone predictions replay exactly. Three schedule pairs retain matched parent states and sampler streams. Completed resume preserves88 artifacts including parent checkpoints, plus the report, with zero new updates.
No learning-rate winner is deployed. Easy percentage is undefined because its stationary CV error is zero.
The supplied annotations include generated/interpolated positions. No seconds, metric, true3D or foundation claim.
A schedule-induced reduction in output jitter is not automatically prediction of future motion.

## Constant-Prediction Sanity Bound

Exactly8566 of15430 training targets are entirely zero (55.52%). At each waypoint the zero fraction exceeds one half. By the triangle inequality, an input-independent offset a increases the summed empirical distance by at least (2*n_zero-n)*norm(a). The smallest per-row coefficient across waypoints is0.120544.
Thus zero is the optimal input-independent path in the stored parent-normalized coordinates on this training population. This is not a constant local decoder code subsequently rotated/rescaled by each observed frame. It does not bound conditional predictors, establish an architectural limitation or rule out useful observed information; it makes output collapse versus actual conditional gain an important distinction.
