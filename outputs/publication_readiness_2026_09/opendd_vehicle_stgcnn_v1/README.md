# OpenDD: a second published trajectory architecture

I adapted the [CVPR 2020 Social-STGCNN core](https://arxiv.org/abs/2002.11927)
to the existing deterministic OpenDD task. This compares predictors
on the same 141,783 histories, 25,596 nonoverlapping queries and
500 recordings, rather than comparing their published scores on
different benchmarks. The two arms share a fitted architecture and
training recipe, with or without the nearest three observed vehicles.
Both use the same focal comoving frame, completion loss and CV
displacement cap as the [EqMotion control](../opendd_vehicle_eqmotion_v1/README.md).

| Forecast | Available ADE | Available FDE | p95 ADE | Conservative gain vs CV (pp; 95% CI) | Original risk passes / 1,500 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CV | 5.161 | 15.183 | 12.497 | Reference | Undefined without edits |
| OLS4 | 6.124 | 16.645 | 14.579 | -27.68 [-29.44, -26.01] | 0 |
| Completion-trained incumbent | 4.104 | 13.715 | 9.720 | +11.54 [+9.27, +13.71] | 14 |
| Social-STGCNN ego only | 6.070 | 15.356 | 13.734 | -39.08 [-44.28, -33.90] | 0 |
| Social-STGCNN with neighbors | 4.701 | 14.620 | 11.292 | +2.33 [+0.80, +3.94] | 3 |

Neighbor input improves observed gain over the ego-only model by
**+23.60 pp** [+21.15, +25.82], but the direct conservative contrast
is **-1.15 pp** [-4.54, +2.24]. Its conservative gain against the
incumbent is **-21.47 pp** [-24.89, -18.05]. Mean signed easy
gain versus CV is +2.25% with neighbors and -99.20% for ego only.
Signed averages do not replace the original selected-risk checks.
I kept the incumbent candidate and the existing CV fallback.

Replaying both frozen backbones on the same rows confirms that
Social-STGCNN is behind the matched EqMotion control: **-62.23 pp**
[-68.79, -56.23] conservative gain for ego only and **-16.66 pp**
[-18.96, -14.31] with neighbors. These are direct paired contrasts,
not differences between summary tables. They qualify the comparison
for this deterministic budget, not for either paper's original task.

Errors use release coordinates. Intervals use 3,000 paired resamples
of seven locations, after averaging the three seeds within location.
The conservative contrast charges distance between the two actual
predictions on missing steps. It is not the difference of their
separate conservative CV gains.

## Reproduce

```sh
dest=data/stage_cvpr2027_experiments/social_stgcnn_author
url=https://raw.githubusercontent.com/abduallahmohamed/Social-STGCNN/ebd57aaf34d84763825d05cf9d4eff738d8c96bb
mkdir -p "$dest"
curl -fLsS "$url/model.py" -o "$dest/model.py"
curl -fLsS "$url/LICENSE" -o "$dest/LICENSE"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_vehicle_stgcnn.py fit
.venv-pytorch/bin/python scripts/run_m3w_opendd_vehicle_stgcnn.py score
.venv-pytorch/bin/python scripts/audit_m3w_vehicle_backbone_contrast.py
```

The loader verifies the author source and MIT license digests before
import. Neither file, a weight, nor a raw trajectory is in this commit.
`fresh_run`: 42 fits and 16,800 optimizer updates on native CPU
PyTorch. Four compute threads, no worker subprocesses, and a
checkpoint with optimizer and sampler every 100 updates. The public
[weight freeze](freeze.json) precedes held scoring (`5ca9444d4`).
`cached_verified`: the original query cohort, observation geometry,
and incumbent/EqMotion assets. Original reference scores reproduce.
`not_run`: external outcome scoring, CREATE training, Stage5C and SMC.

[Protocol](protocol.md) | [Full readout](readout.json) |
[Direct EqMotion contrasts](backbone_pairs.json).
