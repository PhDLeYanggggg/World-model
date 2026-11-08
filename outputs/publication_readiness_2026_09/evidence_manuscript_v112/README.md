# Evidence Manuscript v112

This draft adds the [radius-matched opening comparison](../moving_record_joint_actions_v1/causal_ledger/radius_opening_v1/README.md).
Learned first-edit timing does not beat the matched earliest control.
A second offer adds no outcome-aware joint coverage in this menu.
No deployment change or independent confirmation.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v112
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v112.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v112.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_radius_opening_replay.py
```

Clean source-archive builds matched all eight main and 31 supplementary
pages at 85 DPI with the same separately supplied CVPR style. Main page
7 and supplementary page 31 were visually checked. An anonymous scalar
replay reconstructed six policy rows and 21 source-resampled intervals.
The full coefficient and score replay also matched across all 42 cells.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `862be2bc97069cefb0fae12cde4d334242f4c94159ab3d00315d0dc02a703392` |
| `supplement/supplement.pdf` | `c8515b4af121945747992666aa17ba2a36932e16ea494776b852e0687a883e55` |
| `manuscript_source.zip` | `1f7015b8f4d14f5656473dcc08ee23f227caead1e268de49fb68591363e6d31b` |
| `supplement_source.zip` | `1a7a039f07eb6353b412a71e67987f075f5e752120ce0940dfafc013b79ec532` |
| `moving_radius_opening_replay_v1/anonymous_opening.zip` | `a165d7569316bb5033f71525a01ec03d08698f31509d62386e9fb447b8c88f73` |
