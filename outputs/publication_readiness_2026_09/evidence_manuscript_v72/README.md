# Evidence Manuscript v72

The main paper and supplement now distinguish matched timing capacity,
chronologically payable opportunities, and the performance of a causal
wait controller. The [timing study](../slot_timing_v1/README.md) finds a
substantial hindsight benefit from a two-query delay. The
[release-value study](../release_wait_v1/README.md) trains the wait
comparison directly but does not surpass the frozen anchor. Neither is
independent confirmation or a deployment claim.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v72
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v72.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v72.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The 8-page manuscript and 19-page supplement were built directly and in
fresh directories from their source archives plus the official CVPR style.
All page renders matched at 85 DPI. The modified main page and both
modified supplement pages were visually inspected. PDFs and archives are
local. The paper has not been enrolled or submitted in this work.

| Local artifact | SHA-256 |
|---|---|
| `build/manuscript.pdf` | `ed48a6555057b4753ec0ff56f7ee6936397b058dc69f8e7c35bd2e44ae45fd5e` |
| `manuscript_source.zip` | `8412be77258874d93b65ddf29fc7a1ded37141589cc4b29742d93aa0e5e90304` |
| `supplement/supplement.pdf` | `1ff83d4989c4f11a6b2df301d9d61d0bab1080053c470c07780c8f27d4b45ea7` |
| `supplement_source.zip` | `fa256b83425228c34636c7aa912861a653b2b9ef54f19fe7c0dcd6004bda8c03` |
