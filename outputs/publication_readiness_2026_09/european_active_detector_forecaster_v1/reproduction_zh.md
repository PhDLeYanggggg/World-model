# 当前实际轨迹模型的检测历史实验：复现与恢复

## 本实验在做什么

这是 European SOURCE 开发数据上的真实 Torch 训练，不是旧 SDD t+50 实验。
观察 8 步、预测 12 步，采样间隔为 12 个原始帧，坐标为检测框中点像素。
不报告米、有效秒数、true 3D、foundation 或人工 gold。

当前 cost learner 实际使用的是 grouped temporal / interaction、partial-neighbor、
dimensionless 轨迹模型。之前的 72 模型实验使用 flat complete-neighbor 模型；
那组结果保留，但不能作为“同一实际预测器只改变输入”的对照。

本轮保留全部 9 个实际 parent 设计，每个设计训练 disabled、current、noise、
history 四个输入版本，共 36 个模型；每个模型完整训练 4,000 步。
新增分支只有 512 个零初始化参数，原有输入 mask、部分邻居几何、时空计算、
输出尺度、损失、采样顺序和预算均保持原样。主结果预先固定为 history，
不从读出的 seed、checkpoint 或模型中挑赢家。

## 环境与进度核查

工作目录为 `/Users/yangyue/Downloads/World`，使用原生 arm64
`.venv-pytorch/bin/python`，CPU 4 线程、interop 1、DataLoader workers 0。
不调用硬件资源探测，不启动多进程，不使用 x86_64 Conda / Rosetta。

训练、结果读取和复算共用任务锁。先检查当前 heartbeat 中的 PID，再核查
该进程及最新事件；旧 PID、锁文件或旧日志单独都不证明进程仍在运行。

```bash
cd /Users/yangyue/Downloads/World
cat data/stage_cvpr2027_experiments/european_active_detector_forecaster_v1/heartbeat.json
tail -n 4 data/stage_cvpr2027_experiments/european_active_detector_forecaster_v1/events.jsonl
```

checkpoint 每 200 步原子写入，heartbeat 每 50 步更新。保留模型、AdamW、
采样和 RNG 状态。10 GiB 磁盘预留、1 GiB checkpoint 上限、12 小时边界；
运行慢不是缩减样本、模型或步数的理由。

## 恢复未完成训练

仅在确认旧进程已结束、没有另一个阶段占用同一任务之后运行：

```bash
.venv-pytorch/bin/python scripts/run_m3w_active_detector_forecaster.py train --resume
```

入口核对原注册文件、代码、缓存与已完成 endpoint 的 hash。完成的 receipt
验证后跳过；未完成模型从 checkpoint 恢复。首个 history 模型还需单独完整
重训 4,000 步并比对科学状态。不要删除 receipt 或修改注册文件来绕过检查。
完整 training_freeze 已存在时，训练入口拒绝覆盖，不能当成需要重跑的错误。

## 冻结后的固定读取

先核实 training_freeze 确实含 36 个完整 endpoint、9 个 disabled parent
完全一致和一个完整 history refit 一致，然后提交轻量 receipts 与 freeze。
私有 checkpoint 不提交。固定读取代码和读取注册也必须先提交，才可读取分数。

首次读取依次执行，代码中的提交检查不能跳过：

```bash
.venv-pytorch/bin/python scripts/evaluate_m3w_active_detector_forecaster.py register
.venv-pytorch/bin/python scripts/evaluate_m3w_active_detector_forecaster.py run --resume
.venv-pytorch/bin/python scripts/evaluate_m3w_active_detector_forecaster.py verify
.venv-pytorch/bin/python scripts/report_m3w_active_detector_forecaster.py report
.venv-pytorch/bin/python scripts/report_m3w_active_detector_forecaster.py figures
.venv-pytorch/bin/python scripts/report_m3w_active_detector_forecaster.py bundle
```

`run` 只读取旧 controller-fitting 白名单中、各 producer 未用于训练的 locality
行，并在相同 IDs / batch128 下重新推理实际 parent，要求 disabled 输出完全一致。
保留 149,510 个唯一 query、74 个录像、全部 12 个 locality；不新增读取其余
89 个录像的 controller-held 结果，也不打开独立选择、校准、确认或 DroneCrowd。
这些 SOURCE 数据已有开发暴露，producer-locality 排除不是新的独立测试。

读取包括全部 6 个固定物理基线、history 与三个输入对照、已知标签与缺失结果
completion bounds、hard/easy、完整未来和录像尾部。不按读出的结果选阈值。
3 个真实 seed 与 3,000 次 locality bootstrap 均保留；窗口和重复模型视图
不能当独立样本。locality082 缺最终标签时，完整 roster 的 FDE 不可估，
不能删除该 locality、补零或插值成观测事实。

`verify` 必须在另一个进程重新运行全部 36 组预测、指标与汇总，逐项比对，
不是只检查已有 JSON 的 hash。后续再次验证允许私有 runtime 更新，但报告
保留第一次验证的耗时记录；科学内容或生成代码改变则拒绝覆盖旧证据。

## 公开与私有边界

公开目录：`outputs/publication_readiness_2026_09/european_active_detector_forecaster_v1/`。
私有数据、模型和日志：`data/stage_cvpr2027_experiments/european_active_detector_forecaster_v1/`。
原始输入缓存位于同级 `european_producer_detector_store_v1/`，不公开提交。

完成后公开的 `metric_groups.zip` 仅包含 36 组聚合指标，不包含预测坐标、
原始数据或权重。可执行下面的字节核验和恢复；这属于 cached_verified，
不是 fresh_run 训练、推理或科学重算：

```bash
.venv-pytorch/bin/python -c "from pathlib import Path; from src.evaluation.m3w_metric_bundle import verify; r=Path.cwd(); print(verify(r,r/'outputs/publication_readiness_2026_09/european_active_detector_forecaster_v1',restore=True))"
```

真实重算需要受许可约束的原始数据、缓存和私有 checkpoint。公开代码、
配置、loss 曲线、hash 和 exact replay 不等于匿名从头复现包已经完整。
独立重训使用新的复制实验命名空间，保留旧注册，比较科学状态而非含有
PID / 耗时的 checkpoint 字节。

本实验只检验预测器输入贡献，没有新训练安全切换策略。原有每个必要录像
完整 all/easy 策略 2% 风险要求不变；空选择风险是不可估，不是安全的零。
轨迹预测改善不能自动推出可部署。Stage5C 未执行，SMC 未启用。
