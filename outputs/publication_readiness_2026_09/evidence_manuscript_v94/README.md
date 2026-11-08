# Evidence Manuscript v94

I added the [bounded support timing experiment](../opendd_support_stop_v1/README.md).
The main paper reports the matched timing result as inconclusive.
The supplement separates the causal two-query policy from the
outcome-aware timing-by-recipient capacity table.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v94
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v94.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v94.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py --include-support-stop
```

Local and clean archive builds matched on all 8 main and 26
supplementary pages rendered at 85 DPI, using the same separately
supplied CVPR style. Main page 7 and supplementary pages 25 and 26
were inspected. A clean extraction of the anonymous archive reproduced
99 count tables and 110 intervals across 14 readouts from scalars.
That check does not reproduce training. PDFs and archives remain local.
There was no independent confirmation, promotion or submission.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `29d3e5c5d01d785031cbda36aabfa2b8bf004b7d5ba1434d736de1250b0bdecf` |
| `supplement/supplement.pdf` | `f5d872a80eeacf077f8d7049d7fc0172b151eba62f55b45f7c2c1dfd8055dd1e` |
| `manuscript_source.zip` | `92218ac466405ac24155c988d0d237c893c130361758791a6345e6341adc7fcc` |
| `supplement_source.zip` | `4ebb97e9f5056e13763673fd7c9000d5c2260495096fbf56625a7e1003094c45` |
| `opendd_timing_replay_v7/anonymous_opendd_timing.zip` | `32ae5a2f8a0f6d38d132b54505b027c0b1c3e69146ad590c9bfc59747a8c8c66` |
