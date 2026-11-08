# Working Manuscript v54: Source Candidate Averaging

Section 37 tests whether averaging source candidates repairs the
failure in Section 36. The [v50 main paper](../evidence_manuscript_v50/README.md)
is unchanged. Matching producer averaging did not improve both utility
and safety; the target recordings remain development evidence.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v54
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v54.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

Direct and clean archive builds render identical fifteen-page images.
The added section and references were visually checked. The registered
experiment replayed all nine recordings from the frozen checkpoints,
with zero training updates. No enrollment or submission was made.

| Local artifact | SHA-256 |
|---|---|
| Supplement PDF | `adbe1ec8d60a085f2680c96af774bdd9af17eb691867b383d61e0e1c4ce102f6` |
| Supplement source ZIP | `72d8a2efc2e8981d803fe36fa82c7b0ba970a54ef6f0216ec07a1cca19449459` |

[Experiment and reproduction](../matched_path_ensemble_v1/README.md).
