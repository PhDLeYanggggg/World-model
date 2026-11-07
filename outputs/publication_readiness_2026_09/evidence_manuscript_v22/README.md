# Decision Manuscript And Supplement v22

The six-page anonymous main draft and five-page supplement add the
[constructive sign witnesses](../native_pair_completion_witness_v1/README.md).
The original outcomes and independent evaluation rule remain intact.
This is a working paper, not a submission-ready package.

~~~sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v22
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v22.py main \
  --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v22.py supplement \
  --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
~~~

Both builders check the exact analysis replay receipt. The archives
exclude the official style and all data, weights and trajectories.
Each clean-directory rebuild with the verified CVPR style matches
the local rendering on every page. Source ZIP SHA-256:

- Main: `9533a9bc0706d1ed87d6a510efefcdf3e61502157936ca4d85a39d2338d97bd8`
- Supplement: `2b29c6fad64696dbaf2d91c326624cc793a766fe191f9cc65e3fc0da9323f5e9`

[Aggregate replay](../review_replay_v1/README.md) |
[Chinese local/CREATE guide](../evidence_manuscript_v12/operation_zh.md).
