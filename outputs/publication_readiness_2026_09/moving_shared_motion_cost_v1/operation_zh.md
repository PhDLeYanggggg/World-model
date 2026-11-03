# 共享运动介入对照：中文操作说明

## 本轮检验什么

固定同一个 EqMotion 预测，不再换预测器或扩大网络。两个 cost head 的容量、
初始化、训练行、抽样、预算和损失一致，只比较是否给出额外的 48 维共享
已观测运动摘要。它们包括当前可见、但过去轨迹不完整的其他人的可用运动。
这些不是视频图像特征，也不是已校准的相机位姿。是否有效必须等真实评价。

每个预测器只在一折训练，另外两折分别用作 cost head 训练和评价，再交换
这两个角色。三个种子、三个预测器折、两种角色方向和两个特征版本，共
36 个真正训练的 head。所有标准化只拟合在该 head 的训练折，不能在评价
折重新估计。未来位置、未来有效标签只用于 loss 和评价，不作为输入。

## 何时能运行

本说明写于原始 18 个预测器仍在训练时；新增真实 cost head 仍是 not_run。
入口要求原始预测库全部完成、两个首模型完整重训一致、所有预测重算一致，
不能拿已经跑完的一部分模型提前评价，也不能把合成测试当作真实成功。

使用 `/Users/yangyue/Downloads/World/.venv-pytorch/bin/python`，原生 arm64，
CPU 4 计算线程、1 interop、0 workers。不要切回 x86_64 Conda／Rosetta。
当前这组对照固定 CPU 路径，不宣称 MPS 或 GPU 成功。

```sh
.venv-pytorch/bin/python -u scripts/run_m3w_moving_shared_motion_cost.py register
```

这一步只建立真实数据／预测库／划分／依赖的完整绑定。需要先提交新生成的
轻量 registration.json，再开始下面的训练，避免运行内容与版本记录不同。

```sh
.venv-pytorch/bin/python -u scripts/run_m3w_moving_shared_motion_cost.py pilot
.venv-pytorch/bin/python -u scripts/run_m3w_moving_shared_motion_cost.py train
.venv-pytorch/bin/python -u scripts/run_m3w_moving_shared_motion_cost.py evaluate
.venv-pytorch/bin/python -u scripts/run_m3w_moving_shared_motion_cost.py verify
```

pilot 在三个训练折、两个版本上做 50+50 对照直接 100 步的真实恢复检查，
并估计完整成本。实际预算至少每个 head 2,000 步，而且每个训练 query
必须被完整覆盖至少四遍。批量 72 是当前 6／8／9 个来源组公平抽样所需，
不根据预测分数调大小。慢不构成缩减预算的理由。

## 如何恢复

日志、heartbeat、压缩 checkpoint 和分配状态位于私有目录：
`data/stage_cvpr2027_experiments/moving_shared_motion_cost_v1/`。
先核对真实 PID 和原子 checkpoint，不要因为观察超时就重复启动。
确认旧进程已退出且阶段尚未完成后，原命令增加 `--resume`。完整阶段拒绝
覆盖。恢复会核对角色、数据、模型和随机状态，不允许更换配置后沿用权重。

完整输入矩阵只在 RAM，不保存巨型 feature store。checkpoint、预测库和
分配缓存不入 Git。保留原 10 GiB 磁盘余量；若实际资源不足，保留已完成
工作，核对 CREATE 自己的目录、队列和环境后再转移，不能删除别人的文件
或重复提交正在运行的任务。单次阶段 12 小时边界不等于减小实验目标。

## 怎么判断结果

评价同一预测器下的：不介入、全部用神经预测、逐人选择、整 query 统一
选择、同切换数量联合选择和自由数量联合选择。联合优化只使用预测成本，
不是从真实未来中挑答案。solver 超时或后检查不通过时保留可行的原策略。

完整部署策略仍须满足每个要求录像上的原定 2% operational／easy 风险，
不能只报告平均 ADE，也不能拿同切换数的效用比较代替安全检验。空分母是
不可估，不是安全通过。verify 会重算全部 36 个 head 的输出与评价，并
核对已保存分配的相同输入和原约束；不假称重新求解的限时整数优化一定
逐位一致。两个首 head 的完整独立重训一致性另有收据。

24 个原始来源组中一个没有任务窗口，完整主比较仍不可估；23 个有支持组
上的 3,000 次分组 bootstrap 只是明确标注的开发二级结果。重叠窗口不是
独立样本，来源组件也不是已验证的独立物理地点。正效用、训练 loss 下降
或预测距离缩小均不能单独让风险 gate 或论文准备 gate 通过。

仍然只报告像素和原生 annotation steps，不报告米／秒、true 3D 或
foundation 成功。没有打开额外 Moving 片段、DUT 校准或 DroneCrowd 最终
确认结果；Stage5C 不执行，SMC 不启用。
