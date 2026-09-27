# 本轮实验操作说明

## 这次模型做什么

观察8个历史位置，预测12个未来位置，原始帧间隔为12。它是确定性、有界的
轨迹预测器，不是生成式世界模型。本轮只修复纠偏比例的坐标单位处理，
不增加网络层数、不更换划分、不打开独立测试数据。

九组训练已完成，比较评分有正收益，但相对恒速基线的 easy 误差仍恶化，
不能替换部署。不要把本轮8.46%的整体ADE改善当成历史Stage37的t+50结果。
两个数字对应不同协议，不能混用。

## 本地环境

在项目根目录使用原生arm64的 `.venv-pytorch/bin/python`。本轮实测环境为
Python3.11.1、Torch2.12.0、NumPy2.4.6。入口会在导入Torch前拒绝Rosetta或
x86_64解释器，避免回到旧Conda和Intel运行库路径。

计算线程为4，interop线程为1，DataLoader worker为0；线程数和加载进程数
不是同一设置。不调用GPU/MPS资源探测。不能仅凭import成功判断训练正常，
本轮已有真实100步试跑、九组正式训练和固定第一组完整重训复现。

## 运行与恢复

训练入口和完整顺序见同目录 `reproducibility.md`。训练被中断时，先核实
原进程状态和最后日志，确认已终止再执行：

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_dimensionless_refit.py --phase train --resume
```

它核对身份和数据哈希，恢复优化器、采样器和随机状态；已完成且哈希一致的
端点不会重新训练。每200步原子保存checkpoint，每50步记录heartbeat。
运行慢或暂时读不到日志不等于卡死，不要另开重复写入进程。

私有目录：`data/stage_cvpr2027_experiments/european_dimensionless_refit_v1/`。
`heartbeat.json`记录主进程状态，`events.jsonl`保留历史，各组checkpoint在
`dimensionless/single<fold>_seed<seed>/checkpoint.pt`。保留至少10GiB磁盘。
中断后另存恢复日志，不覆盖旧日志；checkpoint和大缓存不上传GitHub。

## 模型调用边界

模型类是 `DimensionlessAgentTrackSourceForecaster`，输入必须由既有
`pack_geometry`构造，并使用与checkpoint一致的baseline_index及数据schema。
不能把任意二维坐标直接塞入模型，或把米制轨迹当当前图像坐标缓存。
公开runner的predict/replay阶段展示了真实载入和批量推理流程。

已封存预测的精确核查命令：

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_dimensionless_refit.py --phase replay
.venv-pytorch/bin/python scripts/run_m3w_european_dimensionless_refit.py --phase verify_eval
```

重放核对的是同一版本，不等于重新下载原始数据或独立验证。完整重训复现脚本
固定检查第一组4000步；若它的核查目录已有checkpoint，会拒绝覆盖。
这项短时核查不被算作额外独立种子。

## CREATE 使用范围

本轮通过既有授权连接只读检查了队列，未提交、取消或修改远程任务。现有任务
属于simulation项目，不算M3W训练证据；本轮也未新核实M3W远程目录。
本地九组训练进程约9.2分钟、峰值约1.64GB，没有必要转移数据到HPC。

未来规模超过本地合理范围时，先核查项目目录、数据、现有作业和资源限制，
再通过调度器提交，记录jobID、环境、资源、日志、checkpoint和恢复命令。
不在登录节点训练，不凭队列报错推断作业失败，不擅自占用simulation项目的
工作目录或变更其作业。远程暂时不可达时继续独立的本地工作。

## 如何解释产物

`results.md`是固定口径结果；`absolute_costs.md`保留误差尾部和零基线成本；
`failure_analysis.md`列出负结果；`unit_sensitivity.md`是输入缩放检查，
不是泛化准确率；`gradient_diagnostic.md`只比较已记录的训练梯度。
`operations.md`和`verification.json`在完成全部重放后生成，分别记录运行
成本和最终校验，不是部署安全证明。

本轮标签为检测器派生的silver轨迹。坐标、时间均没有验证成米或秒，
不能声称true3D、foundation、物理安全或已达到投稿要求。Stage5C和SMC关闭。
