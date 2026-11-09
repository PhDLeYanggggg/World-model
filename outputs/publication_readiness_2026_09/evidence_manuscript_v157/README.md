# Evidence manuscript v157

The supplement tests whether a past-only estimate of label
availability makes a better decision about deviating from CV and
OLS4. It separates more risk passes near CV from utility against
OLS4, and compares forecast directions at the same per-row peak.
The 27 fresh fits do not yield a qualifying calibration option or
a reliable gain from the learned gate. The paper does not infer
unobserved trajectory errors or claim a new deployment.

The related-work distinction is deliberate: [TrajImpute](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e13a3071bd0aeb97ce41b2da921dfdb6-Abstract-Datasets_and_Benchmarks_Track.html)
introduces missing values in observed ETH/UCY histories. This
experiment uses complete causal histories and handles absent
future evaluation positions without imputing them. TrajImpute's
reused sources do not add new independent source groups here.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v157
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v157.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v157.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` (8 pages) | `811c7f971bd1a835b2453ddfdf978bd1b7a91152b970cd8a1921e98b8fb05624` |
| `supplement/supplement.pdf` (48 pages) | `4613ce9d177d729aacdc19bdcb01b43017913d646fbb388bfb2bf523a997532d` |
| `main_source.zip` | `e76ea2370ab747e7882d21f8ba157905c9c2d5e06f7faa3a7ad502ac035f7030` |
| `supplement_source.zip` | `a3991db20ab1fa886f760873ce712a3982b48b709a7565b083f3e95c594ec1d2` |

All 56 pages matched clean source-archive builds at 85 DPI with
the CVPR style supplied separately. Changed pages were visually
checked. PDFs, archives and checkpoints stay local. Enrollment,
full paper and supplement dates remain November 10, 16 and 23
AoE on the [official schedule](https://cvpr.thecvf.com/Conferences/2027/Dates)
checked November 9. No enrollment or submission was performed.

[Experiment](../moving_reference_bridge_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md)
