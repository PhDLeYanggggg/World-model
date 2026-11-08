# Evidence Manuscript v99

I added the [two-role stopping comparison](../opendd_dual_clock_v1/README.md).
The supplement keeps the learned policy, the post-result deadline
diagnostic and the outcome-aware nomination bound separate. Neither
matched timing comparison supports replacing the original policy.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v99
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v99.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v99.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py --include-dual-clock
```

Clean archive builds and local renders matched on all 8 main and
27 supplementary pages at 85 DPI using the same separately
supplied CVPR style. Main page 7 and supplementary page 27 were
visually inspected. A clean anonymous replay recomputed 142
count tables and 177 intervals across 22 readouts. The checkpoint
recovery separately completed with zero optimizer updates.
These checks do not replace upstream training or independent evaluation.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `f266ef31e93ac80541b40e5eb5c86c06c03945c354ffb6fec0162d97fe7e8de6` |
| `supplement/supplement.pdf` | `439cb72285e2b25f18edf9c982af8c65573cf7dfc52ef5ecd42ae9cd70ddf1d7` |
| `manuscript_source.zip` | `c02b8a2480548e67a13c3e27e485c6876b819c8238860663f1f4b98a4aa17989` |
| `supplement_source.zip` | `e1a83334a1b4962a6a498f0aea1d8905921e1c94eb4f30a2c924fd2f74e17b99` |
| `opendd_timing_replay_v12/anonymous_opendd_timing.zip` | `c2663f4909e7af76c5da56a0eb8786f735b59b3635946a5791d5cf6f4ade1991` |
