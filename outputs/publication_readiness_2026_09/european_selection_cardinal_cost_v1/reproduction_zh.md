# Cardinal Cost 对照实验复现

## 比较什么

本实验不重训轨迹预测器，不增加数据或改变2%风险要求。固定同一个
29,253参数context cost模型，只比较两个训练目标因素：H/EH低估惩罚4vs1，
净增益误差去均值vs保留绝对偏移。原始9个context模型作为已核验缓存对照，
三种新组合各训练9个，覆盖三个SOURCEfold与真实种子17/29/43。

每个cost模型只训练在其预测器未训练过的8个SOURCE场景。每条原query至少
访问4次。模型选择的6个场景已经暴露，只能当开发评价；不是独立最终测试。
calibration、confirmation和DroneCrowd仍关闭。未来标签只能用于SOURCE训练损失
或冻结策略后的评价，不能生成因果输入、选择动作或调整阈值。

## 环境与命令

从项目根目录运行，使用本机原生arm64环境。入口在导入Torch前检查架构。
CPU4线程、inter-op1、workers0；没有DataLoader多进程，没有NumPy训练替代。

```sh
.venv-pytorch/bin/python scripts/run_m3w_selection_cardinal_cost.py register
.venv-pytorch/bin/python scripts/run_m3w_selection_cardinal_cost.py pilot
.venv-pytorch/bin/python scripts/run_m3w_selection_cardinal_cost.py train
.venv-pytorch/bin/python scripts/run_m3w_selection_cardinal_cost.py infer
.venv-pytorch/bin/python scripts/run_m3w_selection_cardinal_cost.py eval
.venv-pytorch/bin/python scripts/run_m3w_selection_cardinal_cost.py verify
.venv-pytorch/bin/python scripts/report_m3w_selection_cardinal_cost.py --export-traces
.venv-pytorch/bin/python scripts/report_m3w_selection_cardinal_cost.py
.venv-pytorch/bin/python scripts/plot_m3w_selection_cardinal_cost.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_selection_cardinal_cost.py tests/test_m3w_selection_cardinal_cost_evidence.py
```

这些是复现顺序，不代表每个阶段已完成；实际完成状态看对应freeze/complete/
verification收据。完成文件不可覆盖，代码或注册约定改变会停止。需要私有
SOURCE/selection数据、原始预测器及其哈希匹配，Git不包含这些大文件。

## 中断与恢复

私有运行目录：
`data/stage_cvpr2027_experiments/european_selection_cardinal_cost_v1/`。
日志、heartbeat、每100步原子checkpoint都在这里。SOURCE初始状态也保存。
健康训练可以持续；每进程12小时上限后按同一注册恢复，不换任务或降低样本量。
中断时保留最后一个完整更新的checkpoint，不保存半步梯度或消耗一半的采样状态。
最多重做上个原子checkpoint之后的更新。

```sh
.venv-pytorch/bin/python scripts/run_m3w_selection_cardinal_cost.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_selection_cardinal_cost.py infer --resume
.venv-pytorch/bin/python scripts/run_m3w_selection_cardinal_cost.py eval --resume
```

先检查PID、heartbeat和实际checkpoint，不重复启动同一工作。文件锁会拒绝并发
写入。本地资源试跑证明可运行后使用本机，不因慢改成quick或假装已完成。

## 怎样读结果

主对照预先固定为mean_absolute减weighted_centered的paired completion-lower
utility。区间按6个locality重采样3000次；fold/seed先在locality内平均。
重叠窗口、重复预测器视图不是独立样本。known-label secondary不能替代主结果。

部署候选还必须在全部必需录像满足原来的all/easy正伤害及easy-net2%要求。
easy参考分母为空时不可估，不算安全通过。保留futureless和partial targets，
不能筛完整future或挑有利场景。观测标签ADE不等于完整12步ADE；endpoint仅diagnostic。

训练loss定义本来就不同，不可把跨组loss数值大小当作公平性能对照；训练首尾也
不是同一批样本，更不是validation loss。readout、原始标量group、实际loss曲线、
独立verify必须一起查看。只有代码和恢复检查通过，不代表科学假设成立。

仍是local-image pixel、obs8/pred12/rawstride12、detector-silver、2.5D轨迹世界状态。
不作metric、seconds、human-gold、true3D或foundation主张。Stage5C和SMC关闭。
