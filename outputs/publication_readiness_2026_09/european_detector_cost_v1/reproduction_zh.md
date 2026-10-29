# 检测框历史代价学习：复现与恢复

## 范围

主任务为 European SOURCE supplied detector-track midpoint，观察8步、预测12步、
间隔12个原始帧。不是旧 SDD raw-frame t50。这里只重训代价头，不重训轨迹预测器。
三个 controller seeds 不是三套独立训练的轨迹预测管线。所有比较仍为原 fitting
子集上的研究证据，不打开独立选择、校准、确认、inner-held 或 DroneCrowd。

## 本机与已有数据

从仓库根目录使用原生 arm64 `.venv-pytorch/bin/python`。CPU4线程、interop1、
num_workers=0；不使用 x86_64 Conda，也不进行设备资源探测。需要本地已审计的
74录像/149,510查询检测历史缓存、原始输入和冻结轨迹/代价模型。这些不在 GitHub。
注册文件绑定代码、输入哈希和原来训练冻结；不要修改冻结文件以绕过校验。

## 正式步骤

```sh
.venv-pytorch/bin/python scripts/run_m3w_detector_cost.py register
.venv-pytorch/bin/python scripts/run_m3w_detector_cost.py pilot
.venv-pytorch/bin/python scripts/run_m3w_detector_cost.py train --resume
.venv-pytorch/bin/python scripts/evaluate_m3w_detector_cost.py register
.venv-pytorch/bin/python scripts/evaluate_m3w_detector_cost.py run --resume
.venv-pytorch/bin/python scripts/evaluate_m3w_detector_cost.py verify
.venv-pytorch/bin/python scripts/report_m3w_detector_cost.py report
.venv-pytorch/bin/python scripts/report_m3w_detector_cost.py figures
.venv-pytorch/bin/python scripts/report_m3w_detector_cost.py bundle
```

注册、真实试跑、完整训练冻结、读出注册都须在各自后续运行前提交。现有已完成
阶段不可覆盖，不要再次启动 pilot/train/run 来覆盖结果；已有完整产物应使用
`verify` 重新推理。这需要原登记的上游私有数据和冻结权重，不代表仅克隆 GitHub
即可从零重建所有资产。独立重训须采用新 replication 命名空间、保留本登记，
比较科学状态而不是含运行时间的压缩 checkpoint 字节，不能用更少录像、窗口、
模型或更新数替代完整设计。本实验尚未交付公开匿名从零重训资产包。

Git 中的288组逐组汇总以 metric_groups.zip 保存，不是原始轨迹数据。如需从
存档恢复逐组 JSON，可用已有校验 API；这只是字节恢复，不是重新计算科研结果：

```sh
.venv-pytorch/bin/python -c 'from pathlib import Path; from src.evaluation.m3w_metric_bundle import verify; r=Path.cwd(); print(verify(r,r/"outputs/publication_readiness_2026_09/european_detector_cost_v1",restore=True))'
```

## 中断恢复

私有路径是 `data/stage_cvpr2027_experiments/european_detector_cost_v1/`。检查
heartbeat.json 中的 PID、UTC、context、representation、bank 和 step，再检查进程
是否存在。正常慢进程继续等待；不要因观察超时启动重复训练。每100步保存原子
checkpoint，每50步写心跳，进程锁阻止重复启动。中断后用同一 `train --resume`
恢复。完整10+10/direct20试跑和第一组2000步独立重训核验均检验科学状态一致；
包含运行时间的压缩文件字节不要求相同。磁盘保留10GiB、checkpoint总额上限512MiB。

## 核验与解释

144模型必须全部2000步，关闭分支的36组必须精确复现旧模型。288评价组不得删除
失败录像或未知标签。training_traces.json 是实际 minibatch loss，不是验证 loss。
summary.json、verification.json、evidence.json、metric_bundle.json 分别提供主统计、
重新推理、执行诊断和聚合文件字节核对。SVG仅含汇总统计，PNG留本地。

完整策略仍要逐必需录像满足原2%风险要求；同切换数对照只比较效用。空风险保持
not_estimable，不当作通过。3,000次bootstrap以12个locality为单位，先平均重复
context和controller seed；不能将重叠查询和444录像视图当独立样本。任何 fitting
提升都不自动构成独立泛化、世界模型成功或可部署结论。

不提交数据、检测框、轨迹、feature store、checkpoint、图像或虚拟环境。SOURCE
仍为像素和原始帧，不宣称米、秒、人工gold、true3D或foundation。Stage5C/SMC不执行。
