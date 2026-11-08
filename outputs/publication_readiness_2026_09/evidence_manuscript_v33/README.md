# Variable Radius and Distributed Motion

Section 16 and Table 13 distinguish matching actual per-target radii
for sparse comparisons from spreading a matched total query budget
over more recipients. The same-half-radius shape comparison is
explicitly post-hoc. The main draft is unchanged.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v33
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v33.py \
  --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

Rebuilding the source ZIP with the same CVPR style produced nine
identical page renders. The new table and disclosure were inspected.
The inherited line-number encoding warning remains.

- PDF SHA-256: `36bf4ae446aefdba7cb69a156b6171e0957d9e48ae82d2c8cb154e30b7f8fd9a`
- Source ZIP SHA-256: `f3c4b7f3232508903818ac82f9d2f74f88bcaea309ed12c667e54bc83c65c128`

[Experiment and reproduction](../opendd_causal_amplitude_v1/README.md) |
[Chinese local and CREATE instructions](../evidence_manuscript_v12/operation_zh.md).
