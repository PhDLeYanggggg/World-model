# Decision Manuscript v15

This anonymous working draft retains the matched-budget and imitation
experiments from [v14](../evidence_manuscript_v14/README.md). It adds
the [dual-reference capacity audit](../native_dual_reference_capacity_v1/README.md)
to distinguish satisfying the recording rule from also improving
conservative utility against both CV and OLS4. These are exposed
development results; this is not an independent or submission-ready
package.

A subsequent [OLS4-blend capacity check](../native_ols_blend_capacity_v1/README.md)
adds no positive views at the same exact budgets. It is documented
separately; the frozen v15 PDF and ZIP are unchanged.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v15
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v15.py \
  --source-zip "$out/manuscript_source.zip"
cd "$out/build"
tectonic -X compile manuscript.tex
```

The builder checks the dual-reference analysis against the witness
replay receipt and retains the v14 source bindings. The archive
contains anonymous source and figures, not the official style,
checkpoints, data, or outcome labels. The
[v14 companion supplement](../evidence_manuscript_v14/supplement_readout.md)
covers the original risk rule, missing-label attribution and source
influence. See the [Chinese operation guide](../evidence_manuscript_v12/operation_zh.md)
for the local and CREATE instructions.

The source ZIP rebuilt in a clean directory with the verified
official style; all six rendered pages matched. ZIP SHA-256:
`c9489d74954d018ac8aedb33f1982f4ca42db514ccd4b869fa943b0e24c2165f`.
