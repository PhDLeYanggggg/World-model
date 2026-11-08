# Evidence Manuscript v111

This draft adds the [same-query support-exchange experiment](../moving_record_joint_actions_v1/causal_ledger/small_support_exchange_v1/README.md).
It separates recovery of easy-risk support from conservative utility
improvement, and gives the radius argument for exchanges that remove
a wholly unlabeled edit. All results are on exposed development.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v111
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v111.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v111.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_small_support_exchange_replay.py
```

Clean source-archive builds matched all eight main and 31 supplementary
pages at 85 DPI with the same separately supplied CVPR style. Main page
7 and supplement pages 30--31 were visually checked. The independent
anonymous scalar replay reconstructed four solver-mode rows and four
intervals. The original risk definition and confirmation roster are
unchanged.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `0c0f2e544e8141b7f5a1a5ffff1a5399fd8bacd924aa6c7b9d6cee55b7714974` |
| `supplement/supplement.pdf` | `a48d0df16f1bda8cf469f3b32ec05649cea172e429b48bfe7e31cde651f64024` |
| `manuscript_source.zip` | `a3625022db219b9a311f165292827cd6218437130d043efc413930196365ec5b` |
| `supplement_source.zip` | `6e6a5e3307b7d1b9e52beb59d86b346d73b398d03181f4513563ba42c5fdf03c` |
| `moving_small_support_exchange_replay_v1/anonymous_exchange.zip` | `fa3ffc3aafd61b6ad171fc84ab0a19917391091e747b815873d8da96f75a1e67` |
