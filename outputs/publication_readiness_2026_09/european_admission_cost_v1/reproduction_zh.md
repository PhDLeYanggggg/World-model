# 本轮复现和恢复

本轮训练的是7候选连续代价头，不是新的轨迹生成器。完整54个实例已训练，
不是 quick、NumPy fallback 或只检查 import。当前结论是修复失败，不部署。

## 环境与本地核验

在项目根目录使用原生 arm64 `.venv-pytorch/bin/python`。CPU计算线程4，
互操作线程1，DataLoader workers0；没有使用MPS或远程GPU。保留至少10GiB
磁盘空间。数据、既有模型、冻结候选缓存和本轮权重需要本机私有资产。
Git仓库仅有代码、配置、汇总指标、训练曲线和报告，不包含这些大文件。

已完成结果的核验命令：

```bash
.venv-pytorch/bin/python scripts/evaluate_m3w_admission_cost.py verify
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_admission_cost_v1 --verify
.venv-pytorch/bin/python scripts/report_m3w_admission_cost.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_admission_cost.py tests/test_m3w_admission_readout.py tests/test_m3w_admission_artifacts.py tests/test_m3w_selected_action_attribution.py
```

`verify`会重新运行全部72组推理、选择和评价，并要求预测、动作、原指标、
支持情况和汇总完全一致；压缩包核验只是字节核验，不是新的科学重算。
报告再生成不训练、不改变阈值，也不代表独立测试。

## 仅对未完成运行恢复

```bash
.venv-pytorch/bin/python scripts/run_m3w_admission_cost.py train --resume
.venv-pytorch/bin/python scripts/evaluate_m3w_admission_cost.py run --resume
```

训练每100步原子保存，每50步写心跳和日志。中断后最多恢复到最近的完整
检查点；不能通过改输入、配置、协议或权重后继续冒充同一运行。完成的
freeze/readout会拒绝重新覆盖，此时应运行verify，不要删除旧结果。
日志和权重位于私有 `data/stage_cvpr2027_experiments/european_admission_cost_v1/`。

本轮已实际验证3种损失的10+10步恢复与直接20步一致；2种旧对照一致；
一个新头完整2000步重训一致；36个完整旧对照一致；另一个进程完成全量
推理重放一致。公开仓库从零下载并重建所有父级私有资产的流程尚未验证，
不能把本机复现等同于公开冷启动复现或独立科学确认。

## 如何读结果

主比较为新损失相对upper_score，而不是相对退化的弱对照。主known差异
-0.075404个百分点，区间跨0；未知整行完成敏感性下界差异为负。easy
违规视图仍有308，未过2%完整策略要求。神经候选的次要正贡献不能替代
这两个失败。35项针对性测试通过，不声称重跑整个历史仓库测试集。
计数中的窗口、种子和旋转有重叠；不能当作独立样本数。没有开启独立
测试、Stage5C或SMC，没有metric、秒级、人工gold、true3D或foundation声明。
