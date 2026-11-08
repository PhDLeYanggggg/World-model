# Working Manuscript v61: Signed Bridge Risk

Section 44 adds the donor-calibrated signed risk comparison and the
admission diagnosis. The [v50 main paper](../evidence_manuscript_v50/README.md)
is unchanged. The refit increased interventions without establishing
a deployable advantage.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v61
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v61.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The direct and clean archive builds render identical sixteen-page
images. The new section and references were visually checked. A
scratch replay made zero training updates, and the admission audit
reproduced exactly. No enrollment or submission was made.

| Local artifact | SHA-256 |
|---|---|
| Supplement PDF | `f6775f2774d3aa8ebb89c998f415979a6fc2fb2173d2daf7363114dc471c7f84` |
| Supplement source ZIP | `9f954323d3d1a1e5debf125e5297d2f3e6603a5f07caa7c231905803ed068d92` |

[Experiment and reproduction](../signed_bridge_risk_v1/README.md).
