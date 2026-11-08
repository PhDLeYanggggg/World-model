# Direct Forecast Candidates in the Supplement

Section 12 and Table 9 on page 7 add the matched OpenDD velocity experiment.
They distinguish a positive development accuracy result from the failed
easy-preservation test and negative interaction ablations. The main draft
is unchanged. No independent confirmation is claimed.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v28
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v28.py \
  --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

Direct compilation and a clean rebuild from the source archive with the
same CVPR style produced seven identical page renders. Table 9 and the
limits paragraph were inspected. The pre-existing line-number encoding
warning does not affect the build.

- Source ZIP SHA-256: `1e255e6f76a4574c04a8fcc212c2ad6779276b78a85bc189b1d6e6d0667c961f`
- Built PDF SHA-256: `106d00bbd2e50bd0a186df7d260309dadb8db77355399412947b36f0ea61aa46`

[Experiment](../opendd_velocity_candidate_v1/README.md) |
[Chinese local and CREATE instructions](../evidence_manuscript_v12/operation_zh.md).
