# Evidence Manuscript v107

This draft adds the [sparse pedestrian intervention](../moving_record_joint_actions_v1/causal_ledger/sparse_support_v1/README.md)
and exact within-class OLS4 bridge. It separates safety and gain
against CV from superiority to both CV and OLS4. The current head
does not close the stronger baseline gap.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v107
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v107.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v107.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_sparse_support_replay.py
```

Clean source-archive builds matched all 8 main and 30 supplementary
pages at 85 DPI with the same separately supplied CVPR style.
Main page 7 and supplementary pages 29 and 30 were visually checked.
The standalone anonymous replay checked 10 policy rows, both
capacity tables, and 22 paired intervals from 14 source groups.
No confirmation outcomes were opened; no submission-readiness claim.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `4a3d49b272948940932f5672a25e064811efba972aaabd963956d44abb91d14d` |
| `supplement/supplement.pdf` | `aed3141a6680cd47ceb62b0edfc2366dc266e7b934fd7379ebae2582aea16334` |
| `manuscript_source.zip` | `f1f80dec1ced591dde97f99c340c7554706039c9b0595341a59cd5b060fdd034` |
| `supplement_source.zip` | `940bec6d974383826b2e71c57176c089efebf9a27f5cd9c612dc44f9701a8929` |
| `moving_sparse_support_replay_v1/anonymous_sparse_support.zip` | `fe51b789d155e51591f9efe6f46b87d153be222a2f275ff79a8ca4d0e2a2b054` |
