# Evidence Manuscript v116

This draft adds [same-query peer inputs](../tumdot_peer_choice_v1/README.md)
and [the exact replacement-count comparison](../tumdot_peer_budget_v1/README.md).
Aligned peers lose in direct conservative utility to both controls,
including after matching replacement counts per query.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v116
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v116.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v116.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_tumdot_peer_replay.py
```

Clean archive builds with the same separately supplied CVPR style
matched all eight main and 32 supplementary pages at 85 DPI. Main
page 7 and supplementary pages 31--32 were visually checked.
Zero-update model replay and the outcome-blind count match agree
with the frozen outputs. The anonymous packet reconstructs 104
descriptive intervals across four comparisons. No new independent
confirmation or submission.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `2067214f6cef235afb50d700e0e03e1c23169738c0fffed8ebeebc075f84454f` |
| `supplement/supplement.pdf` | `bfdf0c4d6691ddff848260be61255f4919219fd0057a145f2b5238dce4c3fcc5` |
| `manuscript_source.zip` | `79794a963a8f365437564d8c144dd6a2173ece41c636e77fe9c550186fcc3afb` |
| `supplement_source.zip` | `5a542bc7d3ad9d3ee7439ab37b669200649a7a293d896cbfa5611e0eba2b58dd` |
| `tumdot_peer_replay_v1/anonymous_peer.zip` | `36a82f2e5f655b20631a3b91651185913ce35467d030b75f6f952c4046c35215` |
