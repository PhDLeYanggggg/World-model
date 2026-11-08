# Evidence Manuscript v108

This draft adds the [pedestrian budget factorial](../moving_record_joint_actions_v1/causal_ledger/budget_risk_v1/README.md),
its matched causal comparison, and the observed/missing utility
decomposition. Wider budgets improve hindsight capacity, not the
learned policy's performance. No confirmation outcomes were opened.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v108
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v108.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v108.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_moving_budget_replay.py
```

Clean source-archive builds matched all 8 main and 30 supplementary
pages at 85 DPI with the same separately supplied CVPR style.
Main page 7 and supplementary page 30 were visually checked.
The standalone anonymous replay verified nine policy rows,
factorial counts and 41 intervals from 14 source groups.
This establishes scalar reproducibility, not submission readiness.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `61a6e525e0c740d948b7d4e62a19c92001173d7af190c83c18b49bab0cf98fc3` |
| `supplement/supplement.pdf` | `7d4a71af6707017fdc0b8008bd01c2e0760eb3f5f8e69fe4da994640d6cbc32e` |
| `manuscript_source.zip` | `ff78d4c089c9d4ec0f68a6643ee52717f4ef4b0c95489d34e468feaa5627b932` |
| `supplement_source.zip` | `aaeee991cff0aee859a5b312b59f3a61f3f241e73467e80f3db62a7b332ca5da` |
| `moving_budget_replay_v1/anonymous_moving_budget.zip` | `07253a98a51c7c2e46afe884b4d53a5933f27a5d4000f054b45edbbdcc491eb2` |
