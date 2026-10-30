# 原始检测历史进入轨迹模型：复现与恢复

## 范围

本实验是 European SOURCE 开发数据上的真实 Torch 轨迹预测实验。
它不是旧 SDD t+50 实验，也不是新独立测试。输入为 8 个过去观测，预测
12 个采样位置，采样间隔为 12 个原始帧；坐标是检测框中点像素。
没有验证秒、米、homography、人工 gold 或实时 sensor-as-of。

72 个模型全部完成并冻结后，才允许固定结果读取。原有 2% 完整策略风险
要求不变；新的轨迹预测更准确，也不能自动推出切换策略安全。独立选择、
校准、确认集与其他 controller-held 结果不在本次读取范围内。

## 环境与正在运行的任务

工作目录：`/Users/yangyue/Downloads/World`。
使用 `.venv-pytorch/bin/python` 的原生 arm64 环境。训练采用 CPU 4 线程、
interop 1、DataLoader workers 0，不调用 MPS/硬件资源探测，不启动多进程。
不要用 x86_64 Conda/Rosetta 替代此环境。

训练首次运行 PID 为 47833。实际当前状态应同时核对进程和有时间戳的日志；
锁文件、旧 PID 或旧 heartbeat 单独都不证明仍在运行。不要启动重复训练。

```bash
cd /Users/yangyue/Downloads/World
ps -p 47833 -o pid,etime,%cpu,rss,stat
tail -n 4 data/stage_cvpr2027_experiments/european_detector_forecaster_v1/events.jsonl
```

主 checkpoint 每 200 步原子写入，heartbeat 每 50 步更新。每个模型保留实际
optimizer、采样和 RNG 状态。已完成模型的公开轻量 receipt 与私有 checkpoint
绑定。完整训练有 12 小时边界、10 GiB 磁盘预留和 2 GiB checkpoint 上限。
本地实测试跑支持完整执行；运行慢不是缩减步数或冒充完成的理由。

## 中断后恢复

只有确认旧进程已退出、没有另一个实验阶段占用同一任务后，才运行：

```bash
.venv-pytorch/bin/python scripts/run_m3w_detector_forecaster.py train --resume
```

入口会检查原注册绑定和输入 hash；已完成 receipt 校验后跳过，未完成模型
从 checkpoint 恢复。不要删除 receipt、覆盖旧 endpoint、修改注册脚本或
为了恢复而放宽样本、划分和预算。已经存在完整 training_freeze 时，不重新训练。
断点后总耗时可能分布在多个 PID 日志中，不能仅用最后一次进程耗时替代总计算量。

## 冻结后的固定结果读取

下列步骤不是当前进程完成的证明。先核实 training_freeze 中确有 72 个
4,000-step endpoint、18 个完整 disabled 对照一致和一个完整 history refit
一致，再把轻量 endpoint receipts 与 freeze 安全提交。私有 checkpoint 不提交。

先固定读取代码，再登记；登记文件提交后才能读取结果：

```bash
.venv-pytorch/bin/python scripts/evaluate_m3w_detector_forecaster.py register
.venv-pytorch/bin/python scripts/evaluate_m3w_detector_forecaster.py run --resume
.venv-pytorch/bin/python scripts/evaluate_m3w_detector_forecaster.py verify
.venv-pytorch/bin/python scripts/report_m3w_detector_forecaster.py report
.venv-pytorch/bin/python scripts/report_m3w_detector_forecaster.py figures
.venv-pytorch/bin/python scripts/report_m3w_detector_forecaster.py bundle
```

`run` 使用旧 controller-fitting 白名单中、各 producer 未训练过的 locality
行；`verify` 在另一个进程重新推理并逐项比对预测、得分和汇总，而不是只检查
JSON 文件能否读取。全部 72 组都要保留，不能选次要分支替换预先指定的主结果。
空分母标记不可估，不当作零风险。

## 文件与可公开复现边界

公开目录：`outputs/publication_readiness_2026_09/european_detector_forecaster_v1/`。
私有数据、模型与日志目录：
`data/stage_cvpr2027_experiments/european_detector_forecaster_v1/`。
完整过去输入另存于 `european_producer_detector_store_v1/` 同级私有目录。

公开配置、代码、hash 和训练曲线可以核对实验设计与已记录工作，但公开仓库
不含 SOURCE 原始数据、缓存和 checkpoint。没有这些资产不能声称新 clone
可以直接完成全部复现。独立从头重训应使用新的复制实验命名空间，保留旧注册，
比较科学状态而非包含 PID/耗时的 checkpoint 文件字节。完整匿名复现包仍需补齐。

本实验不执行 Stage5C，不启用 SMC，也不修改当前部署模型。
