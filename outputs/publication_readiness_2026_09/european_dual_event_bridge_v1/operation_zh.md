# 双风险策略切换实验：运行和恢复

本轮新训练的是介入成本头，不是重新训练 Transformer 或 JEPA 主干。
输入缓存和旧预测器复用并校验；训练、决策和评价单独留有记录。
六个已看过的模型选择场景不能再称独立测试。

## 本地运行

使用项目的原生 arm64 `.venv-pytorch/bin/python`，不要用 x86_64 Conda。
固定 CPU 计算线程 4、互操作线程 1、DataLoader workers 0。数据已在本地，
试跑通过后本轮不需要上传 CREATE；只读队列查询不等于远程训练。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_dual_event_bridge.py --phase register
.venv-pytorch/bin/python scripts/run_m3w_european_dual_event_bridge.py --phase pilot
.venv-pytorch/bin/python scripts/run_m3w_european_dual_event_bridge.py --phase train --resume
.venv-pytorch/bin/python scripts/run_m3w_european_dual_event_bridge.py --phase decide
```

注册文件必须先提交并推送；完整决策冻结文件也必须在评价之前提交并推送。
训练中断时保留原目录，用 `--phase train --resume` 接续，不覆盖检查点。
完整头已存在时验证哈希后复用，不谎称本次又训练了一次。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_dual_event_bridge.py --phase evaluate
.venv-pytorch/bin/python scripts/report_m3w_european_dual_event_bridge.py
.venv-pytorch/bin/python scripts/verify_m3w_european_dual_event_bridge.py
```

## 产物与边界

私有目录：`data/stage_cvpr2027_experiments/european_dual_event_bridge_v1/`。
`heartbeat.json` 保存 PID、当前头和步数；`events.jsonl` 保存长期记录。
`heads/*/checkpoint.pt` 保存模型、优化器和采样随机状态。
`training_complete.json`、`decisions_complete.json` 绑定完整产物。

公开目录保存协议、轻量指标、损失记录、每场景结果、消融、局限和检查报告。
不上传原始数据、缓存或权重。67 个针对性测试文件覆盖当前依赖链；遗留全套
测试没有重跑，不能写成全仓库测试通过。重叠窗口不是独立样本，bootstrap
以六个场景为单位，三个种子先在各场景内汇总。模型选择区间不能当确认性检验。

常规审计和修复由执行流程完成，不再逐项要求用户审计。独立评价边界不放宽；
正式投稿、Stage5C 执行和 SMC 启用仍遵守单独确认限制。
