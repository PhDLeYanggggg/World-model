# Evidence Manuscript v87

I added the [learned timing experiment](../opendd_credit_value_v2/README.md)
to the main paper and supplement. It separates the lack of incremental
gain from a learned continuation value from the loss of hindsight
capacity caused by a fixed recipient rank. Both are exposed OpenDD
development results. The 15/30 DroneCrowd roles and original risk
criterion are unchanged; there is no independent confirmation claim.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v87
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v87.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v87.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The main (8 pages) and supplement (24 pages) were both built in the
workspace and from clean source archives using the same separately
supplied CVPR style. All rendered pages matched at 85 DPI. Main page 7
and supplement pages 23 and 24 were visually checked. PDFs and archives
remain local.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `422dac7b33682aaca86a50a580c06f06147eb6ab7aae71248b0340a90d62ab45` |
| `supplement/supplement.pdf` | `830097a4519b2aa76167cf2dc06291129a3b5ece152bdf71c3946974c2171f2e` |
| `manuscript_source.zip` | `5369828d1dad5c297518460b63e9c5e8bc5893949f74df8b290e2e679525447b` |
| `supplement_source.zip` | `21c5daf4b6759bae0c674b65c78de4a8b4f4390aee4cc1395e5239259976fe3a` |
