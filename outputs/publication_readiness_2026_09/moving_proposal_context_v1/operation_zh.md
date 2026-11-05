# 复现与恢复

本实验只使用已经暴露的 SOURCE/own-B 开发数据，任务是原生 obs8/pred12
图像坐标。没有打开新的保留测试集，没有修改原来的2%风险及支持要求，
没有启用恒等回退认证、Stage5C 或 SMC。

## 运行环境

在项目根目录使用原生 arm64 环境 `.venv-pytorch/bin/python`。
入口在数值库加载前拒绝 Rosetta/x86_64。CPU计算线程4、interop1、
DataLoader workers0，不做硬件探测。训练有进程锁、每100步checkpoint
及heartbeat。无需为这组小型风险头新增 CREATE 作业。

私有文件在 `data/stage_cvpr2027_experiments/moving_proposal_context_v1/`。
不要把数据、缓存、模型权重或运行环境上传 GitHub。

## 新运行与中断恢复

公开协议和代码必须先提交。新实验按顺序执行：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_proposal_context.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_proposal_context.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_proposal_context.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_proposal_context.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_proposal_context.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_proposal_context.py
```

只有某阶段中断且确认进程已经退出，才对该未完成阶段加 `--resume`。
不要根据观察超时重复启动，不要删除旧checkpoint。完成凭据存在的阶段
会拒绝重跑；这是保护已冻结结果，不是需要绕过的错误。

`heartbeat.json` 和 `events.jsonl` 给出PID、阶段、实际更新数及时间。
检查记录在增长和进程活动，再判断是否卡死。使用正常CPU线程慢跑
不是失败理由。没有达到固定训练预算不能写训练完成。

## 已完成结果的只读核验

```sh
.venv-pytorch/bin/python scripts/replay_m3w_moving_proposal_context.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_proposal_context_risk.py tests/test_m3w_proposal_context_reporting.py
```

只读核验不重新训练、不新开科学评价；它重算冻结模型的原有指标并检查
公开文件未变。完全复现不等于独立泛化证据。报告必须区分 fresh_run、
cached_verified 和 not_run，不能把拟合检查通过当作安全通过。

额外 `diagnose_m3w_moving_proposal_context.py` 只分析冻结模型在own-B上的
来源切片与移除bridge后的变化，不做训练、不选模型、不读取新增held
结果。这不是重训消融，不能据此宣称新增动力学能力。
