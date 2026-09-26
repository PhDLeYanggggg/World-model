# 本轮复现说明

## 范围

这是源域开发实验：检验给误差校正器加入被冻结模型自己的风险分数，
是否改善跨 locality 的风险估计。不是重新训练轨迹网络，也不是独立测试、
正式风险校准或部署实验。主协议仍是观察 8 步、预测 12 个标注步，图像像素。
本轮不使用秒、米、物理安全、true 3D 或 foundation 的表述。

原始数据、历史 checkpoint 和逐行分数需要在本机已有，GitHub 只保存代码、
配置和轻量汇总。不能只下载汇总报告就声称复现了训练。

## 环境与执行

使用项目内原生 arm64 `.venv-pytorch/bin/python`。入口会在导入 Torch 前
拒绝 macOS x86_64；计算线程 4、interop 线程 1、DataLoader worker 0。
本轮 CREATE 只作只读队列检查，没有新提交、取消或修改作业。

已经完成的版本优先运行完整验证，不覆盖历史注册文件：

```bash
.venv-pytorch/bin/python scripts/verify_m3w_european_risk_conditioned_residual.py
```

原始执行顺序如下。注册、支持度和预测冻结的提交顺序不可省略，也不可在
读取新的外层结果后倒补注册。旧开发结果的历史暴露不会因注册而消失。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_risk_conditioned_residual.py --phase register
# 提交注册与代码后：
.venv-pytorch/bin/python scripts/run_m3w_european_risk_conditioned_residual.py --phase support
# 核查并提交全部支持度结果后：
.venv-pytorch/bin/python scripts/run_m3w_european_risk_conditioned_residual.py --phase fit
# 提交 prediction_freeze.json 后：
.venv-pytorch/bin/python scripts/run_m3w_european_risk_conditioned_residual.py --phase evaluate
.venv-pytorch/bin/python scripts/run_m3w_european_risk_conditioned_residual.py --phase report
.venv-pytorch/bin/python scripts/plot_m3w_european_risk_conditioned_residual.py
```

## 中断与恢复

本轮是按 view 分块的闭式拟合，不是随机梯度训练。每个完成 view 有模型、
分数和 SHA-256 receipt；重跑 support 或 fit 时校验后跳过已完成 view。
没有完整 receipt 的 view 会重新计算，不能把半成品当结果。
私有目录 `data/stage_cvpr2027_experiments/european_risk_conditioned_residual_v1/`
保存 PID、heartbeat、事件日志及中间产物。文件锁防止重复运行。保留至少
10 GiB 磁盘；不要删除其他任务缓存或 checkpoint 腾空间。

## 读结果

`support_report.json`：144 次拟合行推理和 432 个代数投影检查，不是预测提升。
`prediction_freeze.json`：864 个新探针完成后的哈希，不是 gate 通过。
`aggregate_metrics.json`：全部比较、区间和研究 gate，以它和结论报告为准。
`verification.json`：精确回放、测试及文件封存，仅证明本轮实现可复现。

三种子先在 locality 内平均，然后对四个 locality 配对重采样 3,000 次。
六种 source assignment 彼此重叠，不能算六个独立研究，窗口数也不能当
独立样本数。H_E 是 easy-harm 成本，MSE 的改善不是轨迹 ADE/FDE 改善。
未打开 selection、reserved calibration、confirmation，未执行 Stage5C 或 SMC。
