# Anonymous Decision Supplement

The four-page English companion to the v20 working manuscript
consolidates the original recording risks, paired accuracy and
relative geometry, source influence, matched-policy failures,
the continuous recording feasibility bound, and the latest
past-only motion comparison. It distinguishes the completion
envelope from a confidence interval, and the available forecast
capacity from a policy's actual outcome. The latest tables bind
to four exact development replay receipts. No independent result
was added.

The related-work paragraph compares against
[multi-expert regression deferral](https://proceedings.mlr.press/v235/mao24d.html),
[selective regression](https://proceedings.mlr.press/v162/shah22a.html)
and [multi-expert classification underfitting](https://proceedings.mlr.press/v306/liu26g.html).
Classification identifiability is a possible parallel, not an
explanation established by the trajectory experiments.

## Rebuild

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v20
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v20.py \
  --source-zip "$out/supplement_source.zip"
cd "$out/supplement"
tectonic -X compile supplement.tex
```

The builder uses the existing aggregate review replay ZIP and the
verified CVPR style; it needs no forecast arrays. The source ZIP
contains only the assembled TeX, table values and a short build note.
With the official style restored, a clean-directory build matches all
four page rasters. A repeated assembly produces identical ZIP bytes.
The PDF text, metadata and annotation actions contain no author name,
local paths or external links. This is a scoped artifact check, not
a review-platform or anonymity guarantee. Nine focused tests passed.

Source ZIP SHA-256:
`574a656a8fdae4d42bf51a6760e6bbb45b94c6ace5af93f222a51599df0f4dd7`.

## 中文操作

从项目根目录执行上面的命令；如果 Tectonic 不在 PATH，用桌面应用
附带的 Tectonic。生成的 PDF 和 ZIP 保留在本地，不向审稿系统上传。
匿名 ZIP 解压后只需放入已核验的官方 `cvpr.sty`，即可单独排版。
如要重算实验，而不只是重建排版，先看[聚合复现](../review_replay_v1/README.md)
和[本地及 CREATE 教程](../evidence_manuscript_v12/operation_zh.md)。
补充稿不依赖外部校准标签，重建也不会启动或重启校准。
