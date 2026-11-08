# Working Manuscript v48: Weighted Velocity And Reference Disagreement

Section 31 records the full-strength forecaster experiment and the
triangle lower bound on summed missing-label debit. The
[v45 main paper](../evidence_manuscript_v45/README.md) is unchanged.

~~~sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v48
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v48.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
~~~

A clean archive build reproduced all fourteen page images. The added
section and final reference page were visually checked. No independent
outcomes, enrollment or submission.

| Local artifact | SHA-256 |
|---|---|
| Supplement PDF | `e09a49655fc36b8f6f997c18053d76adc7345878f3b189f07f8eb51e69334be7` |
| Supplement source ZIP | `ca90046b18f2b2b6f51c23b93933226de0e8e225c13340c078cec9ac3088239e` |

[Experiment and reproduction](../weighted_velocity_v1/README.md).
