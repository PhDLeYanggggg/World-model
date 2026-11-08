# Working Manuscript v58: Hannover Transport and Capacity

Section 41 adds the first frozen Hannover transfer, the publisher
citation and the radius/risk capacity analysis. The
[v50 main paper](../evidence_manuscript_v50/README.md) is unchanged.
This readout is not a new official test split or confirmation of success.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v58
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v58.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The direct and clean archive builds render identical sixteen-page
images. The new section and citation were visually checked. The three
clips and geometric limits replayed exactly. No enrollment or
submission was made.

| Local artifact | SHA-256 |
|---|---|
| Supplement PDF | `cf0f5dd07997c555db072bfa67221e69484915aa8513bb4706dd003fa46c5fa7` |
| Supplement source ZIP | `cb2136f3eb85fa727434506044bbd666867fd4a87bd66054ee41a2995450ddb9` |

[Experiment and reproduction](../hannover_frozen_transfer_v1/README.md).
