# Evidence Supplement v76

Adds the [first-credit comparison](../first_credit_v2/README.md) with
full-policy counts, paired intervals and the exact one-edit ceiling.
More development recording checks pass, but neither advance rule
shows reliable utility or qualifies for deployment. The
[main paper v73](../evidence_manuscript_v73/README.md) is unchanged.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v76
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v76.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The 20-page PDF built directly and from a fresh archive extraction
with the same official CVPR style. All 20 renders matched at 85 DPI;
the changed experiment, disclosure and references pages were visually
checked. PDF and source archive are local only.

| Artifact | SHA-256 |
| --- | --- |
| `supplement/supplement.pdf` | `13faa7e623fc02a3517f53e6797ef97a7dc4cdde7b0ce2cedc1a031801e0c5f0` |
| `supplement_source.zip` | `f6971b2a785bb5a3cf75a29aae285411e1ea9ec5a92502d99ecffa3b70ce0a7d` |

No confirmation access or submission. This is a working supplement,
not a submission-readiness receipt.
