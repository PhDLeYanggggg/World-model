# Standalone Review Replay For Development Tables

The local ZIP contains two complete recording-risk and paired-utility
tables (sparse exchange and common motion) and ten joint-path
contrasts. The export derives recording masses and risk ratios from
the actual frozen forecasts, then checks every statistic against the
original score and replay receipts. The reader recomputes the 2%
pass flags, utility contrasts, and 3,000-draw source bootstrap.
It retains all 34 recordings, 14 source groups and three seeds.
The only change to the reported values is the display of seed aliases.

## Verification

- 12 files, 133,659 bytes; SHA-256
  `577bd644da33ad397903529f3cbb7cb1378716a047883284c37a4b68396f43fe`.
  A second export to a separate directory produced the same bytes.
- Extracted into a fresh directory outside the repository and ran
  `python -E -s -B replay.py` with Python 3.11.1 and NumPy 2.4.6.
  All aggregates and intervals matched the frozen tables exactly.
- Checked the file list and scanned text for local paths, account
  identifiers, original source identifiers and weight or cache paths.
  The ZIP has no Git metadata, raw annotations, video, prediction
  vectors or checkpoints. Numeric results are still recognizable
  if compared with a public copy; no anonymity guarantee is made.
- 12 targeted regression tests passed, including hash tampering,
  deterministic packing, and preserving the intervention denominator.

## 中文操作

1. 在项目根目录用 `.venv-pytorch/bin/python scripts/build_m3w_review_replay.py`
   从本地冻结产物重建。默认产物在
   `data/stage_cvpr2027_experiments/review_replay_v1/review_replay.zip`。
   本地已有数据即可完成，不需要重新训练或提交 CREATE 作业。
2. 把 ZIP 解压到独立目录。用 Python 3.11 安装包内
   `requirements.txt`，运行 `python -E -s -B replay.py`。
   加 `--show` 可以查看重算的整张表。
3. 判断是否成功：应报告两张录像表、十组联合对照一致。
   缺失引用分母仍保留不可估状态，不能当作零风险。

The package deliberately contains no link to an author repository or
additional review content. The
[CVPR 2027 author guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)
require anonymity and prohibit external links that expand review
content. This is a local staging artifact, not an upload or a
determination of OpenReview file acceptance.

**Remaining:** raw-data-to-prediction reconstruction, independent
DroneCrowd results, and final anonymous submission QA. This export
does not alter any policy or claim, does not assign external roles,
and does not open any external label.
