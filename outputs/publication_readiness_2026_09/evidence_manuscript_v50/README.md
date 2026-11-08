# Working Manuscript v50: Forecast Shape and Risk

The main text and Supplement Section 33 now distinguish a positive
shape-training result at matched radius from lower recording-level
risk pass counts. This remains exposed OpenDD development evidence.

~~~sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v50
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v50.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v50.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
~~~

Clean source archive builds reproduced seven main and fourteen
supplementary page images. The updated sections were visually checked.
No independent readout, enrollment or submission.

| Local artifact | SHA-256 |
|---|---|
| Main PDF | `ef789980a78b2eabc3daf3bdf886b2a7004fc78e21e54e80983c79932b967bc0` |
| Supplement PDF | `3dcf685081c6771aaee187e21d69532be83251c9b5c64eab578a0351f73ee3bf` |
| Main source ZIP | `79cfc99c56f61b5ee80f3b1cb1e50eecc878db4851052a6b73c95a7b89219823` |
| Supplement source ZIP | `8560788acc4038589493109a008095c59b1990f7b08d5d1a2d19aac1eb2ed02b` |

[Experiment](../opendd_cv_tangent_v1/README.md) |
[Local and CREATE instructions](../evidence_manuscript_v12/operation_zh.md).
