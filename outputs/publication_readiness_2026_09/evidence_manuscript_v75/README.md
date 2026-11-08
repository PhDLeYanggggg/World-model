# Evidence Supplement v75

Adds the controlled [expiry-pricing comparison](../expiry_pricing_v1/README.md).
The asymmetric price reduces expiry loss, but does not establish better
full-policy utility or risk. The [main paper v73](../evidence_manuscript_v73/README.md)
is unchanged.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v75
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v75.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The 20-page supplement built directly and from its extracted source
plus the official CVPR style. All page renders matched at 85 DPI;
the changed page and references ending were visually checked.

| Artifact | SHA-256 |
|---|---|
| `supplement/supplement.pdf` | `8d07df0f68a5c797b6a9eb0695925ae3ed73e8055b2eec863d6c87532cd2d2a1` |
| `supplement_source.zip` | `87750daee8db3116540dcd7f62cf721da80ff1f6edc99553bcf155a6c05f59a0` |

[Standalone statistics replay](../deferral_review_replay_v1/README.md).
