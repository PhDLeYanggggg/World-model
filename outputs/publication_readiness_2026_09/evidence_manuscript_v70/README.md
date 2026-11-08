# Decision Manuscript v70

This working draft adds the matched-recipient and matched-displacement [shape-capacity](../radius_matched_turn_menu_v1/README.md) and [learned shape-selection](../shape_menu_value_v1/README.md) studies. The learned selector improves on the linear edit, but the comparison with the observed-prefix control is inconclusive. The studies use exposed OpenDD development locations. The 15-source-group DroneCrowd calibration and 30-source-group independent-confirmation roles remain fixed; the confirmation groups were not opened. The original 2% rule and the classification of undefined easy risk are unchanged. No deployment or submission-readiness claim is made.

From the repository root, with the local manuscript dependencies and archived inputs available:

```sh
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v70.py main --source-zip data/stage_cvpr2027_experiments/evidence_manuscript_v70/manuscript_source.zip
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v70.py supplement --source-zip data/stage_cvpr2027_experiments/evidence_manuscript_v70/supplement_source.zip
```

The 8-page main manuscript and 19-page supplement built both directly and from independently extracted archives. The resulting page images matched pixel for pixel at 85 DPI. The updated abstract, main text and supplement pages were also visually inspected. PDFs and source archives remain local.

| Local artifact | SHA-256 |
|---|---|
| `build/manuscript.pdf` | `5eb334a99f69e9dd180e2f3fb5f8f34074e54be53ff7218be2b7d7fff6019aaa` |
| `manuscript_source.zip` | `271ea1dc8b7cec2eabcbfb2c4277859bf7c93d21432eeb5423ad9a8077cc6d57` |
| `supplement/supplement.pdf` | `619ba196d4f48302d3216169b5dd745b136c32284eb719857a6cb9a8842c453f` |
| `supplement_source.zip` | `f3928ab40a2c675a6a3dfbf8a31145b7f4d44759dbb5e59c0f5a900219d27be3` |
