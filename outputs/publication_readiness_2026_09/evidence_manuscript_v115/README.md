# Evidence Manuscript v115

This draft adds [three-reference capacity and missing-step attribution](../tumdot_three_reference_v1/README.md)
and the [joint-versus-local teacher ablation](../tumdot_joint_teacher_v1/README.md).
The fixed menu has outcome-aware headroom, but joint supervision does not
improve conservative causal utility on this exposed campaign.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v115
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v115.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v115.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_tumdot_joint_replay.py
```

Clean archive builds with the same separately supplied CVPR style
matched all eight main and 32 supplementary pages at 85 DPI. Main
page 7 and supplementary pages 31--32 were visually checked.
All 54 solver receipts and all 27 trained comparison views replayed;
the anonymous scalar packet reconstructs 98 descriptive intervals.
Neither check opens independent confirmation.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `2a85e0affdb5a206aebb1433035a17d6d8154d01e53e370949de180f91d21593` |
| `supplement/supplement.pdf` | `d74cc355e03ed5f84f90071bdc2a3c27999e811f9c48a448f9e4b942d05c77dc` |
| `manuscript_source.zip` | `c08e15bd49d2d1478e236d6cec029a1f6dba0d77c632a8c01b9e47ed6edb7286` |
| `supplement_source.zip` | `2fa11c26b42b8f7e240f298ccd5a94a5920fa0a207dd8099122fd2ad700cf9e3` |
| `tumdot_joint_replay_v1/anonymous_joint.zip` | `b39dd8f354cbad787f070e5c230a86cee804567c981093aa991955404da5967c` |
