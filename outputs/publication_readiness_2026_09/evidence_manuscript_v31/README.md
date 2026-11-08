# Fixed-Radius Training in the Supplement

Section 14 now distinguishes post-hoc radius matching from training
with the old model's radius in the decoder. It reports the small
forecast gain, the matched negative peer result and the attribution
of recording risk losses to observed easy harm. The main draft is
unchanged; this is still development evidence.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v31
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v31.py \
  --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The source ZIP rebuild using the same CVPR style produced eight
identical page renders. Page 8, the new result and the LLM Usage
Statement were inspected. The prior line-number encoding warning remains.

- PDF SHA-256: `056fdec870c6160b5b8dd0fc4ebd87bdfe4dbff824116ce5467a2a3d64225a1d`
- Source ZIP SHA-256: `9ba8007fa2e54c048fe6adb1e3f756149d7e5fcfc45ab936a8f3f086953cccaa`

[Experiment and reproduction](../opendd_radius_shape_v1/README.md) |
[Chinese local and CREATE instructions](../evidence_manuscript_v12/operation_zh.md).
