# Decision manuscript and supplement v24

The seven-page main draft and five-page supplement now report the completed [DroneCrowd calibration](../dronecrowd_calibration_readout_v1/readout/report.md) and [secondary accuracy and relative-path analysis](../dronecrowd_forecast_diagnostics_v1/report.md). The builder binds both score/replay chains and verifies that no threshold qualified. Neither result uses the sealed confirmation role.

The result supports a limited claim: at the same intervention count, joint selection improves relative geometry over independent selection, but improves little over pair-additive costs. The simple OLS4 trajectory comparator is much stronger, the fixed hard cut barely transports, and the original per-view rule rejects all four gates. The draft does not claim that this is a deployable forecast method, independent confirmation, or a submission-ready result.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v24
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v24.py main \
  --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v24.py supplement \
  --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The anonymous source archives omit the official style, raw data, forecasts and weights. Both clean-directory builds with the verified CVPR style were pixel-identical on every page. The abstract, calibration table and support section were visually inspected. Archive SHA-256:

- Main: `2c70ad1b63a6e3ccd0ef3105317999e9d3d2d3ec730cb2442af3b1a91b7c2614`
- Supplement: `d6dc2fa3da425e65f322c00358825ae2377ad15eee96a8548bbac74543fb0b33`

[Chinese local/CREATE guide](../evidence_manuscript_v12/operation_zh.md) | [Calibration recovery](../dronecrowd_calibration_readout_v1/README.md#local-recovery)
