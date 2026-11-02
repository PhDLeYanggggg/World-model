# Seed-spread 风险输入实验复现

## 本轮改变什么

保持旧mean_absolute cost目标、原444个因果字段、七条候选轨迹、SOURCE划分、
query采样预算和2%风险要求不变。新增16个来自真实三种子预测分歧的描述量，
比较disabled、real、noise。所有组同为30,277参数；新增投影从零开始，原权重
和随机状态一致。disabled必须复现原mean_absolute控制器，不可放宽误差容忍度。

三种子17/29/43的预测器在同一个fold训练于相同四个SOURCE场景，它们都不能
见过该风险头拟合的八个场景。不是把训练过当前场景的模型混进ensemble。
SOURCE原始白名单149,510行；三个fold使用91,562/145,422/62,036行，互相重叠，
不能相加当作独立新样本。selection每fold38,102行，是同一批28段录像的重复视图。

16个输入由预测轨迹和过去尺度产生，不使用观测未来、真实终点、future mask、
central velocity或test goals。不把ensemble均值加入候选动作，不改变轨迹预测器。
noise从SOURCE拟合的8x512条描述量目录抽样，是固定随机先验，不是当前观测。
所有归一化与noise目录只来自拟合SOURCE；selection统计量不参与拟合。

## 本地命令

使用项目根目录下arm64环境，CPU4线程、inter-op1、workers0。入口在Torch
导入前检查架构；不使用x86Conda、DataLoader多进程、资源探测或NumPy训练替代。

```sh
.venv-pytorch/bin/python scripts/run_m3w_selection_uncertainty.py register
.venv-pytorch/bin/python scripts/run_m3w_selection_uncertainty.py features
.venv-pytorch/bin/python scripts/run_m3w_selection_uncertainty.py feature_verify
.venv-pytorch/bin/python scripts/run_m3w_selection_uncertainty.py pilot
.venv-pytorch/bin/python scripts/run_m3w_selection_uncertainty.py train
.venv-pytorch/bin/python scripts/run_m3w_selection_uncertainty.py infer
.venv-pytorch/bin/python scripts/run_m3w_selection_uncertainty.py eval
.venv-pytorch/bin/python scripts/run_m3w_selection_uncertainty.py verify
.venv-pytorch/bin/python scripts/report_m3w_selection_uncertainty.py --export-traces
.venv-pytorch/bin/python scripts/report_m3w_selection_uncertainty.py
.venv-pytorch/bin/python scripts/plot_m3w_selection_uncertainty.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_selection_uncertainty.py tests/test_m3w_selection_uncertainty_evidence.py
```

这是完整执行顺序，不代表所有阶段已完成。实际状态看对应freeze、complete和
verification收据。原始数据、模型、预测、特征和checkpoint保持私有且不在Git。
已完成端点不可覆盖。复核现有标量结果可直接运行report与专项测试；另一次全新
训练应使用新实验登记，不删除旧收据来制造“首次运行”。

## 中断恢复与资源

私有目录：`data/stage_cvpr2027_experiments/european_selection_uncertainty_v1/`。
每100个完成更新保存原子checkpoint，写heartbeat和日志。SIGTERM保留最后完整
checkpoint，不保存半步采样或梯度。恢复最多重做上一个checkpoint之后的更新。
每进程12小时边界与10GiB磁盘余量不改变样本目标；健康训练慢可以继续。

```sh
.venv-pytorch/bin/python scripts/run_m3w_selection_uncertainty.py features --resume
.venv-pytorch/bin/python scripts/run_m3w_selection_uncertainty.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_selection_uncertainty.py infer --resume
.venv-pytorch/bin/python scripts/run_m3w_selection_uncertainty.py eval --resume
```

先核查PID和实际日志，不重复启动。锁会拒绝并发写入。这个规模先本地试跑，
若资源足够就本地正式执行；不能因为CREATE暂时不可用而伪造远程作业状态，
也不为了使用HPC重复提交本地任务。当前对照未向CREATE提交作业。

## 统计与结论

主对照：real减disabled的paired completion-lower utility。负对照：real减noise。
都以同一个SOURCE选择的floor为参照。两项CI下界均需大于0，还必须满足原来
全部录像的all/easy正伤害与easy-net2%要求，才能通过候选gate。

保留所有38,102个target、partial与futureless行；空easy参考不可估，不算安全。
使用六个locality的3000次bootstrap，先在locality内平均fold/真实种子。
重叠窗口不是独立样本。training loss不是validation，不把低loss当作部署证据。
known-label ADE不是全体完整12步ADE；endpoint仅diagnostic。

六个selection场景已经暴露，属于模型开发，不是独立最终测试。risk calibration、
confirmation、DroneCrowd仍关闭。SOURCE拟合表现不是验证。没有新的joint优化，
不能把本轮结果替换成旧matched-joint问题已经解决。

仅是local-image pixel、obs8/pred12/rawstride12、detector-silver、2.5D轨迹世界状态。
不是旧rawt50、metric、seconds、human gold、true3D或foundation。Stage5C/SMC关闭。
