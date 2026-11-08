# Evidence Manuscript v114

This draft adds two TUMDOT development comparisons:
[CV-target shape choice](../tumdot_matched_selector_v1/README.md)
and [direct hindcast replacement](../tumdot_incumbent_selector_v1/README.md).
The guarded paired model improves local joint passes without beating
CV or the fixed hindcast on average conservative utility.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v114
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v114.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v114.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_tumdot_selector_replay.py
```

Clean archive builds with the same separately supplied CVPR style
matched all eight main and 32 supplementary pages at 85 DPI. Main
page 7 and supplementary pages 31--32 were visually checked.
The two model/scoring replays used zero optimizer updates. The
anonymous scalar replay reconstructs 96 intervals across four
study/radius combinations. Three intersections in one exposed
campaign do not constitute independent confirmation.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `7ba141552a477ba0b3aba90ffd9749ef22ee0502c7b2f4e0105b92a369dc2380` |
| `supplement/supplement.pdf` | `982c79699297a468b59f653a9014ca59519394b215c67d4fb39e6fafd88de948` |
| `manuscript_source.zip` | `0ac7eaea917fac059ac2364f09d5932f572a253957a88af0349db7816131bccd` |
| `supplement_source.zip` | `3f70766c7d0b3ac386289943bb3d26ee320c6ea8430973059c259ea5e05afa5f` |
| `tumdot_selector_replay_v1/anonymous_tumdot.zip` | `dbd6135bd3d6be0b936177ac4f68d93e2984f5bdae005d181c8a4c5b7f2f5601` |
