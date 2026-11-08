# Evidence Manuscript v101

This draft adds the [protected gain comparison](../opendd_protected_gain_v1/README.md)
and the fixed-support timing ceiling. It distinguishes reduced easy-reference
loss from a learned utility gain, which was not established. The approved
DroneCrowd 15-group calibration and sealed 30-group confirmation roles and
the original 2% rule are unchanged.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v101
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v101.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v101.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py --include-protected-gain
```

Clean builds from the source archives matched all 8 main and 28
supplementary pages at 85 DPI with the same separately supplied CVPR
style. Main page 7 and supplementary pages 27 and 28 were visually
checked. The standalone anonymous replay reproduced 160 count tables
and 210 paired intervals across 26 readouts. These are reproducibility
checks, not independent confirmation or submission readiness.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `cf45941b11ccdf99fba278847ef8db20f374090430b55b9d0782ce3e14006714` |
| `supplement/supplement.pdf` | `ba8f602cff77f01ef6fa647e3c1cda40cd8bf655ec92eb0aa06223067df36a4a` |
| `manuscript_source.zip` | `886023cdb6dd1e832b073ab1a06e421b34395d3fe481ddab2bea676381aa10cc` |
| `supplement_source.zip` | `b1e3ffe947527a431fc390b86b130bd4e4acb449ae566ee143f3df63d901e047` |
| `opendd_timing_replay_v14/anonymous_opendd_timing.zip` | `c8a48d150659875d1efd144ac9fab64413507cd02c21b23573d677d22cff36b7` |
