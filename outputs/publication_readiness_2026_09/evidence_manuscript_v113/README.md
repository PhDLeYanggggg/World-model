# Evidence Manuscript v113

This draft adds the [fixed-radius first-offer ranking comparison](../moving_record_joint_actions_v1/causal_ledger/opening_slate_v2/README.md).
Some diverse slates improve recall, but no ranker chooses a qualifying
first action. Listwise training loses to the matched pointwise model.
No deployment change or independent confirmation.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v113
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v113.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v113.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_opening_slate_replay.py
```

Clean archive builds with the same separately supplied CVPR style
matched all eight main and 31 supplementary pages at 85 DPI. Main
page 7 and supplementary page 31 were visually checked. Full replay
matched all 42 source/seed cells without training; the anonymous
replay reconstructs five policy rows and 18 intervals.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `c282d5c6bc4709bdc837f0a335a5084a7b28e47f02c6ac8756505b036c344dc1` |
| `supplement/supplement.pdf` | `0d8d59497213bf974571a3a1fdeff9a67f2ad964a57221d39eeddb43b9a8c5b2` |
| `manuscript_source.zip` | `0407de5cf0075913e93ab87467d0688768361f573548ff234956b9cac5c05a3e` |
| `supplement_source.zip` | `e0f347bb3ffb8141cef1da7cc18120dfaaba5155557583390fff470fa3eba55a` |
| `moving_opening_slate_replay_v2/anonymous_slate.zip` | `02beaf81cd5990d714acf473b60aab7f8c3b8065a3d0f6fe1033794e3c382eeb` |
