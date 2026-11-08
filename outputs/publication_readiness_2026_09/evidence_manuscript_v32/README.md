# Recording Risk and Fixed-Radius Feasibility

Section 15 and Table 12 add the matched 84-fit experiment and a
closed-form optimistic bound for fixed-radius recording risk. The
bound distinguishes missing reference from unavoidable unknown debit
and hindsight-feasible cases. It cannot certify a causal policy.
The main draft is unchanged.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v32
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v32.py \
  --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The source ZIP rebuilt with the same CVPR style produced nine identical
page renders. Page 8, the new table and the formula were inspected.
The inherited line-number encoding warning remains.

- PDF SHA-256: `a109a2493ba60b79674f5818ef4e36777f17824709e1bc6dadb412a0a2368696`
- Source ZIP SHA-256: `fd84e8f9910326e88aaa73ff2e093deefc6d7ee1e3186c5ccb4a740882feb877`

[Experiment and reproduction](../opendd_flight_shape_risk_v1/README.md) |
[Chinese local and CREATE instructions](../evidence_manuscript_v12/operation_zh.md).
