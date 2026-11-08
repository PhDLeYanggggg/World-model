# Working Manuscript v46: Matched Geometry And Allocation

Section 29 of the supplement compares timing, direction, and recipient
capacity on development sources. The [v45 main paper](../evidence_manuscript_v45/README.md)
is unchanged. The evidence is not independent confirmation.

~~~sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v46
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v46.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
~~~

A clean source archive rebuild reproduced all thirteen supplementary
page images. Section 29 was inspected visually. No enrollment or
submission was made.

| Local artifact | SHA-256 |
|---|---|
| Supplement PDF | `cb5cae2e30cfcae603f67451710c0185e7c8895038842ded18deb3bd666d5e23` |
| Supplement source ZIP | `7514d1afe6295b7f5f53ed62c3a8e179ad00491e01068079b86a0f48de99a62b` |

[Direction and allocation experiment](../matched_direction_v2/README.md) |
[Timing control](../query_temporal_envelope_v1/README.md).
