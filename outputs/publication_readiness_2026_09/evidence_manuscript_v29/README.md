# Recording-Support Evidence in the Supplement

Section 13 and Table 10 on page 7 combine the recording feasibility result,
the streaming joint-control comparison, and the equal-switch easy-support
ablation. They distinguish missing reference, defined risk failure and
conditional usefulness. The main draft is unchanged and independent
confirmation remains sealed.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v29
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v29.py \
  --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

Direct compilation and a clean archive rebuild with the same CVPR style
produced seven identical page renders. The new table and proof paragraph
were inspected. The existing line-number encoding warning remains.

- Source ZIP SHA-256: `55af6dac036dceb8809ba3ecd609c5a9bc87c22580774d68b035cf69aa0041d0`
- Built PDF SHA-256: `719a7141465f54d7ce8da8301098452c43b790d4bbe65d794243cf7d7275cd0b`

[Easy-support experiment](../opendd_easy_support_v1/README.md) |
[Chinese local and CREATE instructions](../evidence_manuscript_v12/operation_zh.md).
