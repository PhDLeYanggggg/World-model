# 联合近邻成本对照：复现与结论

本轮是非参数成本估计，不是新训练神经网络。因此没有新增 optimizer loss；
前一轮 72 个 Torch 成本头的真实 loss 曲线仍保留在 completion-cost 实验目录。
不能把本轮的近邻推理、旧权重读取或程序通过称为新的 Torch 训练成功。

## 实际运行

使用原生 arm64 `.venv-pytorch/bin/python`、CPU4、workers0。进程不做设备探测，
不启动 DataLoader 多进程。先对真实完整首个 context 中 128 个查询做重复检索，
再运行全部 288 组：6 个原始 context、每个 4 个 source、4 个固定模式、3 个
旧成本头对照 seed。新检索策略本身是确定性的，不是三个新训练 seed。

```bash
.venv-pytorch/bin/python scripts/run_m3w_coupled_local_cost.py register
.venv-pytorch/bin/python scripts/run_m3w_coupled_local_cost.py pilot
.venv-pytorch/bin/python scripts/run_m3w_coupled_local_cost.py run --resume
.venv-pytorch/bin/python scripts/run_m3w_coupled_local_cost.py verify
.venv-pytorch/bin/python scripts/report_m3w_coupled_local_cost.py
.venv-pytorch/bin/python scripts/plot_m3w_coupled_local_cost.py
```

首跑 PID33535，260.627765 秒，峰值 10,343,464,960 B；独立重放 PID34342，
258.380133 秒，峰值 10,829,643,776 B。重放重新计算全部近邻及结果，而非只核对
缓存指标。全部计算结束；不是等待一个未知状态的训练进程。

私有路径 `data/stage_cvpr2027_experiments/european_coupled_local_cost_v1/` 保存
每条原始录像的近邻检查点、heartbeat、日志和 PNG 预览。中断时先核实 PID 和
日志，再用 `run --resume` 恢复同一研究范围；不要复制任务或删除结果绕过保护。
有 12 小时边界、10 GiB 磁盘余量和 512 MiB 本实验邻域缓存上限，不因慢而缩小任务。

## 数据和因果边界

只用原始 fitting packets。每个查询排除整条自身录像的 donor 标签；不读取
inner-held、独立选择、校准、确认或 DroneCrowd。查询特征为过去和因果预测。
测试验证了改变整条查询录像的标签不会改变该录像的预测。但上游模型和原始
normalizer 曾见过 fitting cohort，所以不能称为全流程 OOF 或独立泛化。

四种模式是近邻 known-only、近邻 completion endpoints、全局 completion 均值和
记录内整体打乱已知目标。原始未知标签在评价中仍是 NaN，保守端点不是实际误差。
参考成本跨七个动作共享，伤害和收益按因果 envelope 解码，并保持必要大小关系。
大小关系正确不等于选中风险经过校准。

## 结果怎么读

主模式 local_bounds 的已知 ADE 增益是 +0.830158%，但完整风险只支持 30/444
个重复录像视图，仍有 273 个已知 easy 风险超标视图，不部署。与旧 known-only
MLP 的 completion-lower 差异区间跨零。神经候选相对物理候选有很小的拟合正
增量，但不能越过总体风险失败，也不能称作独立验证或 world-dynamics 成功。

`readout.md` 保留全部模式、物理对照、区间、hard/easy 和未知标签统计。
`metric_groups.zip` 是 288 个聚合指标 JSON 的压缩包，不包含原始坐标、权重或
视频；恢复它不等于重新计算科学结果。公开仓库不提供私有输入和预训练权重，
干净 clone 不能在缺少这些资产时直接训练或复现。

## 针对性测试

```bash
.venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_coupled_local_cost.py \
  tests/test_m3w_coupled_local_readout.py \
  tests/test_m3w_coupled_local_shuffle.py \
  tests/test_m3w_completion_cost.py \
  tests/test_m3w_completion_cost_readout.py \
  tests/test_m3w_multiaction_cost_readout.py
```

该范围 24 项通过，不代表历史全部套件重跑。另有保持动作可用性分层的打乱
诊断，按独立的 `stratified_protocol.md` 及 v2 登记执行；它不替换原始 primary。

这是 SOURCE 斜视摄像头 detector-silver 的 pixel/raw-frame obs8/pred12，不是
SDD human gold、米、秒、true 3D 或 foundation。Stage5C、SMC 均未执行。
