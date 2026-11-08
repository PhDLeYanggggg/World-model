# Evidence Manuscript v110

This draft adds the [matched attribution and direction experiment](../moving_record_joint_actions_v1/causal_ledger/incumbent_direction_v2/README.md).
It distinguishes observed utility, missing-label disagreement, and
recipient changes. At fixed recipients, a paired ADE head still loses
to the incumbent. No policy promotion or independent confirmation.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v110
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v110.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v110.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_incumbent_direction_replay.py
```

Clean archive builds with the same separately supplied CVPR style
matched all eight main and 31 supplementary pages at 85 DPI. Main
page 7 and supplementary pages 30--31 were visually checked. The
standalone anonymous replay reconstructs five policy rows and 75
intervals from 14 source groups. Frozen model replay required zero
optimizer updates. These are reproducibility checks, not independent
scientific confirmation.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `ba27a23d233b94863c34b5ce27abb728ddc4c147a61ccb7e4597e07800c4634e` |
| `supplement/supplement.pdf` | `279aca421804a91b644c0616e52ff298cbcb936eec1a3b679228ee955b2f7a3f` |
| `manuscript_source.zip` | `f4c589998adcd0e44a379cca0310ca5d9784e4a6e768d79fcb02710d7979111a` |
| `supplement_source.zip` | `511e25ccec243d79f5d56f2ffd391349a05c05fd4f2c9f4f74920a7371064c37` |
| `moving_incumbent_direction_replay_v2/anonymous_direction.zip` | `bb58f409ea4fa6c2a91c3a3bc71422cc6193325b0767103d4223c18e7420cebf` |
