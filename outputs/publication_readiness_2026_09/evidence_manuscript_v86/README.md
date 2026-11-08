# Evidence Manuscript v86

I updated the main manuscript and supplement with the
[neural edit timing experiment](../opendd_release_credit_v1/README.md).
The result separates the extra schedules available to a hindsight solver
from the gain realized by a policy that only sees the current and past
queries. This is exposed OpenDD development evidence, not an independent
test. The 15 DroneCrowd calibration groups and 30 sealed confirmation
groups remain separate; the original 2% safety rule is unchanged.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v86
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v86.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v86.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

Both sources built in the workspace and from clean archive extractions
with the same CVPR style. All 8 main and 23 supplementary pages matched
when rendered at 85 DPI. Main page 7 and supplementary pages 22 and 23
were visually checked. Generated PDFs and archives remain local.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `cea5ef6729629d3f306d8dd9b669735124974478d653299275bf7e94067bac8e` |
| `supplement/supplement.pdf` | `73124212bbf9d00a8828bcf8be35919a67377e40cc3cc37e9b87df5bd0ca9532` |
| `manuscript_source.zip` | `e9d7bf35ef71071d0e1070abc8b2dc82728d5f9464f4bba94eed16f1697e33a2` |
| `supplement_source.zip` | `eb7c570a28262e48f704904bad52f2e227fb17bd9d72647b1cf496275cdbe348` |
