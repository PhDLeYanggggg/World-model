# Evidence manuscript v158

The manuscript adds a real camera-input comparison from the
[WEPDTOF release](https://vip.bu.edu/projects/vsns/cossy/datasets/wepdtof/download/).
On the same 20 matched targets, OLS4 beats CV with camera
histories and loses with annotation histories. This is a
one-camera development diagnostic. The other 13 sources are
reserved for one-shot external evaluation and have not been
scored. The new study is not a neural dynamics or risk pass.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v158
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v158.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v158.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` (8 pages) | `6d1c395a65802a0ed5b3a99705969298103c880ccc70e95df51f8e4b657c0246` |
| `supplement/supplement.pdf` (48 pages) | `b7d38e9ecdbc454497955a947f7617c70a556ad9a8994d0ee104922b176c08bd` |
| `main_source.zip` | `27908f8159ca1be3a0db62b48f8db3ffa8dac9af13823618d348f17df4d549d0` |
| `supplement_source.zip` | `7daddaee1a1ae02edd5aeb5511aa4fabd1e90c72e47eff27274aacffb5e9a756` |

All 56 pages matched a clean source-archive build at 85 DPI with
the CVPR style supplied separately. Changed pages were visually
checked. The [official schedule](https://cvpr.thecvf.com/Conferences/2027/Dates)
checked November 9 lists registration on November 10 and the
full paper on November 16 (AoE). No registration or submission
was performed. The PDFs, ZIPs and camera data remain local.

[Experiment](../wepdtof_source_probe_v1/README.md) |
[Chinese reproduction](../../../reproducibility/M3W_CURRENT_ZH.md)
