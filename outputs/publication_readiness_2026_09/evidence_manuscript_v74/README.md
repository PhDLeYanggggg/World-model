# Evidence Supplement v74

This version adds the [peer ablation and expiry replay](../wait_peer_value_v1/README.md)
to the supplement. The aligned peer representation does not beat the
self-only and wrong-focal controls reliably. Replaying the exact emitted
slots separates utility on moves from the cost of dropping immediate
edits. The [v73 main paper](../evidence_manuscript_v73/README.md) is unchanged.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v74
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v74.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The 19-page supplement built directly and from its freshly extracted
source archive plus the official CVPR style. Page renders matched at
85 DPI and the changed page was visually checked. These local artifacts
do not contain independent confirmation results.

| Artifact | SHA-256 |
|---|---|
| `supplement/supplement.pdf` | `57459c7354594d0687513cf31d36761cf713f99d179b101a10a83cb9f36a12ad` |
| `supplement_source.zip` | `ee0c23fcca2f846a243dbb36592e91e81503c919e2ae1881b55b80ff2596eefd` |
