# Working Manuscript v57: Learned Road Context

Section 40 compares no-map, real-map and simultaneous wrong-map heads,
then perturbs map inputs with the trained weights held fixed. The
[v50 main paper](../evidence_manuscript_v50/README.md) is unchanged.
The map-sensitive decisions do not establish a map-alignment gain.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v57
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v57.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The direct and clean archive builds render identical sixteen-page
images. The new section and reference pages were visually checked.
The held-location experiment replayed from scratch with zero training
updates. No enrollment, submission or confirmation was performed.

| Local artifact | SHA-256 |
|---|---|
| Supplement PDF | `5cb072280b710aca88da0778258081d5b8b9e38ae954bfc5f32967ff05965f2d` |
| Supplement source ZIP | `507733fee19e65552329805efc76c696845b463522ebe69f1d68b445a1987ffb` |

[Experiment and reproduction](../opendd_map_path_value_v1/README.md).
