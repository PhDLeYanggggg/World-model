# Evidence Supplement v78

Adds the [matched two-edit experiment](../two_edit_v1/README.md)
with exact one- and two-edit capacity, equal-count causal controls,
and the negative relative-path comparison. The
[main paper v73](../evidence_manuscript_v73/README.md) is unchanged.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v78
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v78.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The 20-page PDF compiled in the workspace and in a fresh archive
extraction with the same official CVPR style. All 20 page renders
matched at 85 DPI; the changed page, disclosure and references were
visually checked. The PDF and archive are local only.

| Artifact | SHA-256 |
| --- | --- |
| `supplement/supplement.pdf` | `84db1463761be4cc2ae07f8f5bb61ed29e84204142a5d2ac4289d37e97bc4468` |
| `supplement_source.zip` | `d7a892c779ebf8314652831a41f040a01642ff698082359a1a417b78459ac45a` |

This is a development result, not confirmation, submission or readiness.
