# Decision-First English Working Draft

This three-page CVPR-style extract consolidates the causal decision
evidence. It is a working draft, not a replacement for the fuller
[v10 draft and supplements](../evidence_manuscript_v10/README.md)
and not ready to submit. No external score or admissibility is implied.

| Claim in the draft | Frozen source |
| --- | --- |
| Same-count independent, scene, additive and joint comparisons | [Transport candidate](../dronecrowd_transport_candidate_v1/analysis.json) |
| Recipient expansion and strong-comparator capacity | [Fixed-capacity refinement](../native_recipient_budget_capacity_v1/refinement.json) |
| One recording fails at all seeds | [Easy-reference case](../native_easy_support_certificate_v1/analysis.json) |
| Hindsight retrieval and actual screen-first policy | [Retention](../native_recording_witness_retrieval_v1/retention.json), [portfolio](../native_joint_path_portfolio_v1/analysis.json) |
| Literal and material history cap results | [History radius](../native_history_radius_v1/analysis.json) |
| Different external roles and confirmation prerequisites | [Calibration](../dronecrowd_calibration_readout_v1/README.md), [confirmation](../dronecrowd_confirmation_readout_v1/README.md) |

The working extract keeps CV as the risk reference and OLS4 as an
accuracy comparator. It distinguishes unweighted selected risk masses
from query-weighted accuracy and does not treat overlapping windows
or forecaster seeds as independent recordings.

## 本地重建

先运行 `.venv-pytorch/bin/python scripts/plot_m3w_decision_bottlenecks.py`
重建矢量图。在本地构建目录使用 `decision_core.tex`、已核验的
CVPR review `cvpr.sty` 和生成的图（在构建目录名为
`decision_bottlenecks.pdf`），用 Tectonic 编译即可。
构建不读取原始数据、不启动训练，也无需 CREATE。
已检查的本地 PDF 位于
`data/stage_cvpr2027_experiments/evidence_manuscript_v11/decision_core_build/decision_core.pdf`。

[TeX](decision_core.tex) | [Build receipt](build_verification.json)
