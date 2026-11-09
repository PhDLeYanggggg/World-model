# Evidence manuscript v141

The supplement reports the variable-count query scorer, matched
independent and no-pair controls, and the outcome-aware menu
feasibility result. The main text is unchanged from
[v140](../evidence_manuscript_v140/README.md).

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v141
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v141.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
|---|---|
| `supplement/supplement.pdf` (42 pages) | `7b1eb7ac928689177fb4efbcaafa576cf03295b8892339bf806afdb9e9102c54` |
| `supplement_source.zip` | `e12a493eb068817d4db9551af87dc724d26f51a3c3c98334140368d2b5b9b040` |

All 42 page renders match a clean source-archive build at 85 DPI
with the separately supplied style. Pages 41 and 42 were inspected.
The PDF and source archive remain local.

The [CVPR 2027 dates](https://cvpr.thecvf.com/Conferences/2027/Dates)
list enrollment November 10, full paper November 16, and
supplement November 23, 2026 (all AoE). No enrollment or submission
action was taken. This experiment does not make the paper ready.

[Experiment](../moving_cv_joint_count_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md).
