# Expert Allocation Supplement

This six-page supplement extends the [v24 draft](../evidence_manuscript_v24/README.md)
with two replay-verified fitting-source experiments: the motion
mixture and the fixed-count expert allocation comparison. The main
paper is unchanged. The new table preserves the negative matched
comparisons and keeps hindsight, fitting, calibration and independent
confirmation separate. This is not a submission-ready result.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v25
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v25.py \
  --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The source archive rebuilt from a clean directory with the same
verified CVPR style. All six rendered pages matched, and the new
table on page 5 was visually checked. The bundled line-numbering
package emits an encoding warning; the table fits the column.
The anonymous archive contains only the TeX and a short README,
not raw data, forecasts, weights, or the style file.

- Archive SHA-256: `a2a85ec7aa54e88ad415011972c72a249b2ccf82108a64c8cf4a40633e83e615`
- Built PDF SHA-256: `477d2006cf634c80f97f21fe617e51dfb6682006e22ff6a647e32864c819170d`
