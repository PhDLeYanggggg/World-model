# Evidence manuscript v160

This version adds a development-only diagnostic on the exposed
WEPDTOF camera. At the frozen edit count and radius, the optimistic
upper bound rules out catching OLS4 on known steps. The
[experiment](../wepdtof_camera_capacity_v1/README.md) separates that
limit from the original 2% harm constraints. The 13 scored external
sources were not reopened. There is no new deployment or confirmation.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v160
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v160.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v160.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The 8-page main paper and 49-page supplement matched their clean
source-archive rebuilds at 85 DPI. The changed pages were inspected.
The PDFs and source ZIPs are local.
