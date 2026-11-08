# Main Draft and Context Supplement

The main abstract and Section 9 now include the matched OpenDD
movement and support evidence. Supplement Section 18 gives the new
past-context ablation and same-query shuffled control. This separates
development, DroneCrowd calibration, and sealed confirmation. The
results do not establish deployable or independently validated utility.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v35
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v35.py main \
  --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v35.py supplement \
  --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The two source ZIPs were rebuilt in clean directories with the same
CVPR style. All seven main and ten supplement page renders match.
The abstract, new main table, supplement table and disclosure were
visually inspected. The inherited line-number encoding warning remains.
The archive member check excludes data, weight and video files. Extracted
PDF text was checked for personal workspace paths and author-name tokens.
This is a manuscript rebuild, not a cold reproduction of all training.

| Artifact | SHA-256 |
| --- | --- |
| Main PDF | `e3074b3b925f2606644bc10f0aaedef7a91a80206ff279f5b0ff9b2f9f3a93f3` |
| Main source ZIP | `cc86dbfae39ced055f819dcdcbe969d43615847a613054775ce86c3eb75e2f2a` |
| Supplement PDF | `854d366e94d636bdca0f3cc3d042ba810a3aff39186760392b58413ba4598e4c` |
| Supplement source ZIP | `7ebb56823a771f94fed5faf7f6ebf29186b40ccfc3645c2e29a4bc582ad5bfb0` |

[Context experiment](../opendd_context_exchange_v1/README.md) |
[Chinese local and CREATE instructions](../evidence_manuscript_v12/operation_zh.md) |
[Official CVPR 2027 deadlines](https://cvpr.thecvf.com/Conferences/2027/Dates).
Registration is November 10, full paper November 16, and supplement
November 23, all AoE. The seven-page main contains the required
disclosure under the [author guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines).
No registration, upload, submission, or readiness claim is made here.
