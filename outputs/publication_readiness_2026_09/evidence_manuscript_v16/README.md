# Decision Manuscript v16

The [v17 draft](../evidence_manuscript_v17/README.md) adds the
matched neighbor-context and relative-geometry comparison.

This six-page anonymous working paper extends [v15](../evidence_manuscript_v15/README.md)
with two matched-count development diagnostics: the
[OLS4-centered bank expansion](../native_ols_blend_capacity_v1/README.md)
and the [risk/utility debit ablation](../native_missing_debit_capacity_v1/README.md).
The original 2% rule, external split, confirmation access, and deployment
status are unchanged. There are no independent outcomes in this draft.

The subsequent [speed and turn comparison](../native_damped_motion_v1/README.md)
is documented separately and does not change the frozen v16 files.
The follow-up [trained motion-head experiment](../native_gated_motion_residual_v1/README.md)
and [optional-menu audit](../native_gated_motion_menu_v1/README.md)
are also exposed development evidence, not part of the frozen draft.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v16
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v16.py \
  --source-zip "$out/manuscript_source.zip"
cd "$out/build"
tectonic -X compile manuscript.tex
```

The builder checks both aggregate results against their replay receipts.
The source ZIP excludes the publisher style, data, predictions, and
weights. An empty-directory rebuild using the verified style produced
identical rasters for all six pages. Source ZIP SHA-256:
`49069dfc3e30f9a52f6848519c1ba161041c89c28e06b31fe36bd29e32d2b05f`.

[Companion supplement](../evidence_manuscript_v14/supplement_readout.md) |
[Chinese local/CREATE guide](../evidence_manuscript_v12/operation_zh.md).
