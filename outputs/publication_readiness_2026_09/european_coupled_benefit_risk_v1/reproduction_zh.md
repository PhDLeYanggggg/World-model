# 联合收益/伤害风险头：本地运行、恢复与核验

## 本轮是什么

本轮确实训练了神经网络：36个风险头、72000次主更新、两次2000更新完整复跑。
不是新的轨迹预测器、JEPA 表征预训练或世界动力学训练。两个分支都使用原有
462维过去特征和固定预测轨迹；只比较冻结收益与释放收益约束，参照误差预测
以及原定2%正伤害预算保持不变。未来只参与监督和误差评价，不进入推理输入。

原始输入、归一化、预测器和旧完整控制策略是 cached_verified。首次训练及
完整评分发生在2026-10-30，是该实验的 fresh_run；2026-11-01继续读取这些
产物属于 cached_verified，新进程完整 replay 才是本轮 fresh 重算。原始评分
的旧 session 已不可用，但72组收据和终止标记都存在，不伪造新观察到的退出码。

数据为观察8步、预测12步、间隔12 raw frames 的欧洲斜视摄像头 detector-
silver 图像局部坐标。不是 SDD t+50，不是秒、米、human gold、true3D 或
foundation。独立校准和确认数据未打开，Stage5C/SMC 均禁止。

## 环境和真实资源

在 `/Users/yangyue/Downloads/World` 使用原生 arm64 的 `.venv-pytorch`，
不用 Intel Conda 或 Rosetta。CPU计算线程4、interop1、DataLoader workers0，
不做 Torch resource probing，不启动加载子进程。真实400次更新 pilot 验证
了训练、两分支精确恢复和旧模型控制，而不只验证 import torch。

pilot 计时18.008531秒、峰值9.59GB；完整训练计时333.204713秒、峰值10.98GB。
计时从 runtime_ready 开始，不包含最初 admission/加载。完整评分计时
1086.630145秒、峰值11.11GB。没有因慢减少样本或训练步数，本机48GB足够。
本轮未提交 CREATE、未取消或重启远程任务，也不声称当前远程队列已核实。

## 首次运行顺序

以下为首次实验流程，已有完成结果不会被覆盖：

```sh
.venv-pytorch/bin/python scripts/run_m3w_coupled_benefit_risk.py register
# 提交科学代码、配置、协议及 registration 后才开始自己的真实训练。
.venv-pytorch/bin/python scripts/run_m3w_coupled_benefit_risk.py pilot
.venv-pytorch/bin/python scripts/run_m3w_coupled_benefit_risk.py train
.venv-pytorch/bin/python scripts/report_m3w_coupled_benefit_risk.py --training-only
mkdir -p data/stage_cvpr2027_experiments/european_coupled_benefit_risk_v1/progress
.venv-pytorch/bin/python scripts/run_m3w_coupled_benefit_risk.py eval
.venv-pytorch/bin/python scripts/run_m3w_coupled_benefit_risk.py verify
.venv-pytorch/bin/python scripts/report_m3w_coupled_benefit_risk.py
```

完整模型先全部冻结，再完成72组评分，最后读取主要结果。verify 重新推理、
分配和评分所有72组；它继续使用已经核对的旧控制策略，不谎称重新求解旧
完整控制。共同切换数量的次要控制会重新计算，不能替换完整策略安全要求。

第一次 eval 因缺少私有 progress 父目录退出，尚未产生新评分。原 traceback
保留在 readout.log。仅创建目录后用相同 eval --resume 继续，成功运行日志为
readout_after_progress_setup.log。没有改科学定义、模型或阈值。目录预检是
必要步骤，不能声称原入口会自动创建这个目录，详见 execution_note.md。

## 健康任务和中断恢复

```sh
pgrep -fl run_m3w_coupled_benefit_risk.py
cat data/stage_cvpr2027_experiments/european_coupled_benefit_risk_v1/heartbeat.json
tail -10 data/stage_cvpr2027_experiments/european_coupled_benefit_risk_v1/train.log
```

用实际 PID、日志更新和恢复点一起判断，不能因一次观察超时就重复启动。
每50更新有 heartbeat、每100更新有 checkpoint；分配每250查询及每组完成时
有恢复点。健康运行允许到12小时，保留10GiB磁盘和1GiB压缩 checkpoint 上限。
只有确认进程终止、阶段未完成后，保留日志并恢复对应阶段：

```sh
.venv-pytorch/bin/python scripts/run_m3w_coupled_benefit_risk.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_coupled_benefit_risk.py eval --resume
```

两条分别对应失败阶段，不是一起运行。verify 中断可重跑完整 verify，保留
第一次评分；它不更新模型、不调阈值，不算新的独立科学样本。

## 当前产物的公开核验

```sh
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_coupled_benefit_risk_v1 --verify
.venv-pytorch/bin/python scripts/report_m3w_coupled_benefit_risk.py
.venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_coupled_benefit_risk.py \
  tests/test_m3w_coupled_benefit_allocation.py \
  tests/test_m3w_coupled_benefit_readout.py \
  tests/test_m3w_coupled_benefit_artifacts.py
```

bundle 只核对公开的72组汇总 JSON 字节；report 还需要已登记的私有风险头
checkpoint、完整训练 freeze 和 replay 收据，不能假装公共 ZIP 包含了这些。
完整逐行复算另需原始数据、特征及旧模型，它们不提交 Git。artifact 检查要求
真实完整 readout/replay 和报告已经存在；缺前提时不能声称通过。

## 负结果如何解读

释放收益约束降低部分固定拟合损失，但完整策略效用下降。已知标签增量
-0.543227pp，主要 completion-lower 增量-11.080186pp，同数量控制也为负。
原定完整风险仅6/444视图通过，362个已知 easy 风险违规，不能部署。

未知未来上的预测分歧不是未知行数，少切换不保证稳健区间更窄。缺 endpoint
的完整 FDE 保持不可估；空 selected-risk 分母也不可估，不补成安全零。
这不表示完全不干预的 baseline 会增加真实伤害，而是条件风险缺乏分母。
12个来源已用于开发，3000次 bootstrap 只是条件性名义区间，三个 critic
种子不是三个独立轨迹预测器。训练跑完、损失下降、完整 replay 通过均不代表
核心假设成立、独立泛化成功或论文已达到投稿质量。

后续若需要 CREATE，先核对现有任务、迁移后的环境和数据位置，再按调度器
提交，记录 job ID/checkpoint/日志；本说明不授权重复旧作业或绕过已有数据角色。
