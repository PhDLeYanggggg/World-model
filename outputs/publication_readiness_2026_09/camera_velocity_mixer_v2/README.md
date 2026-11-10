# A stable loss did not make the camera forecast deployable

I repeated the nine matched velocity-mixer fits from the
[original run](../camera_velocity_mixer_v1/README.md). The only training
change was the loss denominator: it now has a floor of 0.31455, the
10th percentile of positive RMS step sizes in the donor histories.
I kept the observed features, seeds, 800 updates per fit, and the
existing CV/OLS4 references and 2% checks. The maximum recorded batch
loss fell from 771,838 to 2.21; the saved losses use the two different
normalizers and should not be compared as measures of accuracy.

| Forecast | Observed ADE (px) | Observed FDE (px) | Conservative gain vs OLS4 (pp) | Easy positive-harm upper |
| --- | ---: | ---: | ---: | ---: |
| OLS4 | 14.61 | 24.72 | 0.00 | 27.30% |
| Median8 | 13.14 | 22.07 | +3.72 | 41.11% |
| Noise spline (prior) | 13.48 to 13.73 | 22.48 to 23.08 | +2.49 to +4.36 | 29.67% to 33.84% |
| Clean mixer | 15.47 to 16.56 | 26.97 to 29.06 | -11.32 to -7.37 | 15.95% to 32.33% |
| Noise mixer | 14.00 to 14.46 | 23.78 to 25.31 | -2.06 to -0.32 | 38.73% to 46.19% |
| Noise + easy penalty | 13.37 to 13.54 | 22.21 to 22.38 | +4.12 to +4.51 | 15.12% to 21.75% |

The easy-penalized refit beats OLS4, median8 and the prior spline on
conservative utility in each matched seed, but not median8 on ADE or
FDE. No learned arm passes the 2% checks. The protected arm's signed
easy error actually improves by 6.45% to 7.75%; that average does not
offset the separate positive-harm bound on harmed rows. A lower ADE
or better signed easy score alone cannot justify release.

This is one previously exposed WEPDTOF camera, with 58 histories and
57 complete futures. The three seeds quantify fit sensitivity, not
independent camera replication. The one-shot external readout and CV
fallback stay unchanged. No new camera threshold or deployment was
selected. The normalization repair does not add the ability to predict
movement from a perfectly stationary prefix.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/run_m3w_camera_velocity_mixer.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_camera_velocity_bank.py tests/test_m3w_wepdtof_frozen_capacity.py
```

Use the native arm64 interpreter; four compute threads and no workers.
The entry point checks registration hashes, writes a checkpoint and
heartbeat every 100 updates, and resumes automatically. A second run
loaded all nine checkpoints and reproduced the frozen predictions and
scores. Ten tests passed. The original code is preserved at `b31242a8b`.

[Protocol](protocol.md) | [Training and forecast freeze](freeze.json) |
[Per-seed scores](readout.json). No non-pilot external outcomes were read.
