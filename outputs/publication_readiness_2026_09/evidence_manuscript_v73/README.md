# Evidence Manuscript v73

The paper now separates two findings from the [whole-recording deferral
experiment](../ledger_wait_v2/README.md). Training on actual credit and
expiry does not establish an advantage over local supervision. But the
actual decoder has unused, count-preserving opportunities when allowed
one label-aware change per recording. The study is OpenDD development
evidence, not independent confirmation or a qualification decision.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v73
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v73.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v73.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The 8-page main paper and 19-page supplement built directly and from
freshly extracted archives plus the official CVPR style. All renders
matched at 85 DPI; the changed main and supplement pages were also
visually inspected. PDFs and ZIP files remain local.

| Local artifact | SHA-256 |
|---|---|
| `build/manuscript.pdf` | `4b19e747bc4759c183cbec15fd1cbaf3f5cb320c3ed3bc744769961f7643e7db` |
| `manuscript_source.zip` | `c6416fd624d586dbb40703df2c9b4bbbd32d4650f440376dd9c69a20c74534b3` |
| `supplement/supplement.pdf` | `ca1ca2b8a4ee625a9e9d375c5addc223544174f1cf3c01c650262ed0360fbc95` |
| `supplement_source.zip` | `afcdfb7ffc411de1fd41b368362053dd5075ae25eb9e0046425b66781b8d45ea` |
