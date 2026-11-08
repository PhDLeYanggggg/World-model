# Evidence Manuscript v109

This draft adds the [pedestrian observability experiment](../moving_record_joint_actions_v1/causal_ledger/observability_v1/README.md).
Better prediction of future label availability steers edits away from
unlabeled tracks but does not improve conservative decision utility
at matched counts. No independent confirmation outcomes were opened.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v109
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v109.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v109.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_moving_observability_replay.py
```

Clean source-archive builds matched all 8 main and 30 supplementary
pages at 85 DPI with the same separately supplied CVPR style.
Main page 7 and supplementary page 30 were visually checked.
The standalone anonymous replay (Python with NumPy) verified six
policy rows and 23 intervals from 14 source groups. This establishes
scalar reproducibility, not independent confirmation or submission readiness.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `ce7f182b96a12bd2f72a231cd3e6ddda64ac0f97e2ebe7dd79a4a2cfeceeaf4f` |
| `supplement/supplement.pdf` | `4978f3567f219694b8dc5176ca61531896ac247f98d0124f88f66cb1c1e17aca` |
| `manuscript_source.zip` | `86bbe3d33884c6d8894779454d80974a15e8cb7442ec4b86de98a9f0b7be281a` |
| `supplement_source.zip` | `94ce404dad3d95ec3c1e49c1fdb2ff14466279bfcc9aa0fac68e6bee86a6f6f0` |
| `moving_observability_replay_v1/anonymous_observability.zip` | `2a04cf952ba452a24d426c4bb5f44d6362feafe28c1a8e8f47627ebe72dfc032` |
