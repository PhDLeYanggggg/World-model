# 配对成本训练：本地复现说明

## 这次实际做了什么

真实训练 72 个小型 Torch 成本头，每个 2,000 次更新，共 144,000 次主更新。
另有 2,000 次完整重训校验和 180 次恢复试验更新。没有新训练轨迹预测器。
两组使用相同的因果查询流，区别是是否把未知结果的保守区间端点计入损失。
原始未知标签仍是未知，不把端点称为实际误差。

主训练耗时 296.70 秒，峰值约 10.18 GB；首次读出 185.26 秒，独立进程精确重放
188.51 秒。固定区域诊断 78.97 秒，重放 80.74 秒。所有计算均已结束，不需要
保留等待中的训练进程。时间是本机实测，不是未来机器的保证。

## 环境

在项目根目录运行，使用原生 arm64 的 `.venv-pytorch/bin/python`。入口在导入 Torch
前拦截 Rosetta/x86，CPU 计算线程 4，interop 1，DataLoader workers 0。
不启动 multiprocessing，不做设备资源探测。当前规模适合本地，不需要重复提交
CREATE 作业。CREATE 访问或队列未知不能当成远端作业已失败。

缓存、输入轨迹、预训练预测器和 72 个成本头权重均只在本地私有路径保存。
公开 Git 仓库只有代码、配置、聚合结果、哈希及统计 SVG，不能仅靠公开仓库
重建缺失的第三方数据或私有权重。新环境必须先按资产来源恢复它们。

## 已注册的实际运行链

```bash
.venv-pytorch/bin/python scripts/run_m3w_completion_cost.py register
.venv-pytorch/bin/python scripts/run_m3w_completion_cost.py pilot
.venv-pytorch/bin/python scripts/run_m3w_completion_cost.py train --resume
.venv-pytorch/bin/python scripts/evaluate_m3w_completion_cost.py register
.venv-pytorch/bin/python scripts/evaluate_m3w_completion_cost.py run --resume
.venv-pytorch/bin/python scripts/evaluate_m3w_completion_cost.py verify
.venv-pytorch/bin/python scripts/diagnose_m3w_completion_regions.py register
.venv-pytorch/bin/python scripts/diagnose_m3w_completion_regions.py run
.venv-pytorch/bin/python scripts/diagnose_m3w_completion_regions.py verify
```

这些命令有冻结与重复运行保护，不是可覆盖历史结果的脚本。现有主读出和诊断
已经完成，`run` 可能拒绝覆盖；只核验现有结果时使用 `verify`。训练与读出共用
进程锁，不能同时启动重复任务。不要删除锁文件或缓存来绕过已有运行。

## 中断与恢复

每 100 次更新原子保存 checkpoint，每 50 次写 heartbeat。恢复命令为
`scripts/run_m3w_completion_cost.py train --resume`，它核验配置、输入与已有状态。
私有目录是 `data/stage_cvpr2027_experiments/european_completion_cost_v1/`。
先检查其中 heartbeat、日志及对应 PID，再决定恢复；不能只因界面无输出就重启。
12 小时边界、10 GiB 磁盘余量和 512 MiB 本实验 checkpoint 上限是显式保护。
不会因为慢而自动改成 quick 或减少科学目标。

## 报告与损失曲线

```bash
.venv-pytorch/bin/python scripts/report_m3w_completion_cost.py traces
.venv-pytorch/bin/python scripts/report_m3w_completion_cost.py report
.venv-pytorch/bin/python scripts/report_m3w_completion_cost.py support
.venv-pytorch/bin/python scripts/plot_m3w_completion_cost.py
```

`training_traces.json` 包含全部 72 头的记录点，图为每种监督的 36 头损失中位数和
10%–90% 描述范围。两组目标和有效监督不同，不能凭绝对 loss 更小判断模型更好。
公开图是程序生成的统计 SVG，不包含原始视频截图；PNG 预览保留在私有目录。

完整重点测试命令：

```bash
.venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_completion_cost.py \
  tests/test_m3w_completion_cost_readout.py \
  tests/test_m3w_completion_regions.py \
  tests/test_m3w_completion_reporting.py \
  tests/test_m3w_scene_context_cost.py \
  tests/test_m3w_multiaction_cost.py \
  tests/test_m3w_multiaction_cost_readout.py \
  tests/test_m3w_metric_bundle.py \
  tests/test_m3w_forest_fitting_diagnosis.py \
  tests/test_m3w_selected_cost_diagnosis.py
```

本轮该范围 34 项通过，不代表历史全部测试套件已重跑。

## 如何核对结论

主结果在 `readout.md`、`summary.json`、`verification.json`；机器摘要在
`evidence.json`，固定区域诊断在 `fitting_cost_summary.json`，受限策略支持上界在
`support_ceiling_note.md`。`failure_analysis.md` 区分实际结果与未证明的机制解释。
哈希记录不是独立场景验证。

结论是失败：混合策略增益从 +0.581% 变为 -18.512%，完整风险支持从 77/444
降到 26/444。损失有记录、训练真实完成、重放精确，但这些都不等于科学 gate
通过。独立选择、校准、确认集及 DroneCrowd 没有打开；不部署新策略。

本任务是 SOURCE 斜视摄像头检测轨迹的 pixel/raw-frame obs8/pred12，不是
SDD human-gold、米制或秒级预测，不是 true 3D 或 foundation。Stage5C、SMC 均未执行。
