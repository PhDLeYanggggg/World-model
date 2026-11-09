# Evidence Manuscript v132

The new supplement section separates annotation-availability accuracy,
matched-count utility, easy-risk failures, and the late-taper control.
None of the variants qualifies on calibration. The figure from v131
is retained with its original source bindings.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v132
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v132.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v132.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
|---|---|
| `build/manuscript.pdf` (8 pages) | `b9d6ab1d2b59e374f40cfcddf5652fb518e464200d3c0a570f9f083636591c0d` |
| `supplement/supplement.pdf` (38 pages) | `d0330a216a1a56970b609bde78b9b82d4201ea6054122d53e26af88274cb8aa7` |
| `main_source.zip` | `f5dd76c1597a2e65fae002c8e9673e653051479ab733b790be22d18f44f89b97` |
| `supplement_source.zip` | `4e9a3192def9ec2f11897f97bfd8e1ae567bae94811ec5fd5b52fdac0a6853cb` |

All 46 pages match clean source-archive builds at 85 DPI using the
same separately supplied style. The new supplement text was inspected
visually. The separate anonymous archive reproduces the primary
scalar risk and utility table. No PDFs, archives, raw data or
weights are committed. Not submitted or independently confirmed.

[Lead-risk experiment and replay](../moving_cv_lead_risk_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md).
