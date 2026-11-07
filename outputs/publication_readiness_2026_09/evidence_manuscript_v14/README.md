# Decision Manuscript v14

This six-page anonymous working draft distinguishes the existence of
feasible forecast replacements from recovering them using past
observations. It adds an exact-budget figure and the matched structured
imitation failure to [v13](../evidence_manuscript_v13/README.md).
It positions imitation against decision-focused learning without
claiming a differentiable solver, a new surrogate loss, or a safety
guarantee. It contains no independent results and is not a submission
package.

The figure and evidence macros are bound to the audited
[exact-count capacity](../native_exact_count_joint_capacity_v1/README.md),
[risk-shortfall](../native_recipient_shortfall_v1/README.md) and
[structured imitation](../native_structured_imitation_v1/README.md)
results. The [v12 supplement](../evidence_manuscript_v12/supplement_readout.md)
remains a separately verified companion for risk definitions and
source influence. The [Chinese operation guide](../evidence_manuscript_v12/operation_zh.md)
covers local and CREATE reproduction.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v14
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v14.py \
  --source-zip "$out/manuscript_source.zip"
cd "$out/build"
tectonic -X compile manuscript.tex
```

The default builder expects the verified official style already
available locally; pass `--style` to point at a separate copy. The
11-file source ZIP excludes the style, author identifiers, local
paths, data, and weights. Compilation in a clean directory with the
verified style reproduced all six page rasters. This verifies the
typeset artifact, not end-to-end training. Local ZIP SHA-256:
`d5ab3558776e88b286c80b401cb415117f71af32489863a9c31936aa5d11295e`.
