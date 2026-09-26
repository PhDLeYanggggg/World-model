# 本轮操作与复现

这是一轮拟合场景上的离线误差诊断，不训练新模型，不读取新的外层评价，
不改变部署。旧预测仅在行号、目标、schema 和哈希校验后复用。
日常审计与同步由项目执行流程负责，以下是接续记录，不是用户待办。

## 运行

使用原生 arm64 `.venv-pytorch`。入口在导入数值库前拒绝 macOS x86_64。
计算线程 4、interop 线程 1、workers 0，无 DataLoader 多进程。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_cap_diagnostic.py --phase register
# 注册代码、配置和 protocol 提交后：
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_cap_diagnostic.py --phase pilot
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_cap_diagnostic.py --phase run
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_cap_diagnostic.py --phase report
.venv-pytorch/bin/python scripts/plot_m3w_european_fixed_cap_diagnostic.py
# 汇总及解释报告齐备后：
.venv-pytorch/bin/python scripts/verify_m3w_european_fixed_cap_diagnostic.py
```

## 恢复与检查

私有目录 `data/stage_cvpr2027_experiments/european_fixed_cap_diagnostic_v1/`
保存 PID/heartbeat、事件日志、每个视图的结果和哈希凭据。
文件锁防止重复运行。中断后重跑 run 校验后跳过已完成视图；未完成的视图
重新计算，不把部分产物当完成。verify 才是显式重算全部视图的复现检查。
保留至少 10 GiB 磁盘，不清理其他任务数据。慢而仍有进展时继续等待。

`completion.json` 证明 144 个视图完成，不代表任何模型通过提升门槛。
`summary.json`、两个 CSV 和图表保留全部口径，不选最有利的一组。
`verification.json` 证明精确回放及针对性测试，不证明独立泛化或部署安全。

## 解释限制

projection floor 使用真实未来标签，只可离线诊断，不进入推理。
它不是不可约噪声、模型实际可达的提升或条件偏差证明。常数条件均值也
可能有较大的逐样本投影 floor；测试中包含这个反例。
三种子和多个视图有依赖，不按窗口数制造显著性。本轮不新增 bootstrap
显著性主张；先前正式比较的场景配对区间保持原样。
观察 8 / 预测 12 个原生标注步，检测器图像像素。禁止 metric、seconds、
true 3D、foundation、human gold 或物理安全宣称。Stage5C、SMC 均关闭。
