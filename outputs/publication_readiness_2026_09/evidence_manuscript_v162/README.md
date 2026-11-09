# Evidence manuscript v162

This version adds the [camera training and capacity experiment](../camera_noise_control_v1/README.md).
It distinguishes noise-trained forecasting, the existence of a
full-size action plan under the original risk test, and the still
untested ability of a past-only selector to recover such a plan.
The grey print figures from v161 remain. The introduction cites
raw-video tracking, history denoising and detection-based forecasting;
the experiment is not presented as a reproduction of those methods.
The independent confirmation set stays sealed.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v162
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v162.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v162.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` (8 pages) | `5fa30d73f66ade46e5f91ca62b69810f808d3deb602f2e83459b3840173e18d6` |
| `supplement/supplement.pdf` (49 pages) | `c79061d3758e0f22b8060383e6454bfebbbd2d15cead412998e20433acb47e6d` |
| `main_source.zip` | `020e9a6bc8e0eea06f79466c4ca8a907e3480511d20d15cb8f344cb253b0409e` |
| `supplement_source.zip` | `d59248593c29432fe4cd1b9a9196ab2053b0b46fa87ebd5baea94180f6f31c4a` |

A clean source-archive build matched all 57 rendered pages at
85 DPI, with the CVPR style supplied separately. The new main text
and supplement pages were inspected. PDFs and source ZIPs stay local.

The [CVPR 2027 schedule](https://cvpr.thecvf.com/Conferences/2027/Dates),
checked November 9, lists registration on November 10, the paper
on November 16, and supplementary material on November 23 (AoE).
No registration or submission was performed. The causal selector
and independent evidence gaps remain; this is a working manuscript.

[Chinese reproduction](../../../reproducibility/M3W_CURRENT_ZH.md)
