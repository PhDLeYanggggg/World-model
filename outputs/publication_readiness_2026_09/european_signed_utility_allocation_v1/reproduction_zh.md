# 本轮实验的运行、恢复与结果核验

## 结果来源

本轮不重新训练神经网络。18个固定风险头、候选轨迹和 source 数据是
cached_verified；重新推理、联合求解及误差统计是 fresh_run。独立校准、
独立测试、新动力学训练和部署均为 not_run。不能将本轮称为神经训练成功。

数据任务为观察8步、预测12步、间隔12个 raw frames，使用图像局部坐标和
detector-silver 标签。不是秒、米、人工金标准、true3D 或 foundation 结果。
Stage5C execution 与 SMC 均禁止。

## 本地环境

在项目根目录使用原生 arm64 的 `.venv-pytorch/bin/python`，不用默认 Intel
Conda 或 Rosetta。入口在导入 Torch 之前检查架构。计算线程4、interop1、
DataLoader workers0，不做 Torch resource probing，不启动加载子进程。

真实 pilot 已验证推理和求解、中断恢复与直接运行一致；这比仅能 import
Torch 更强，但仍只是工程验证，不是模型收益证明。本轮实测内存在本机48GB
范围内合理，因此没有提交 CREATE 作业，也没有重启或取消旧远程任务。

## 原始执行顺序

```sh
.venv-pytorch/bin/python scripts/run_m3w_signed_utility_allocation.py register
# 科学代码、配置、协议和 registration 提交后，才运行自己的实验。
.venv-pytorch/bin/python scripts/run_m3w_signed_utility_allocation.py pilot
.venv-pytorch/bin/python scripts/run_m3w_signed_utility_allocation.py run
.venv-pytorch/bin/python scripts/run_m3w_signed_utility_allocation.py verify
.venv-pytorch/bin/python scripts/report_m3w_signed_utility_allocation.py
```

已完成的阶段拒绝覆盖。完整实验必须有72组，不能将128-query pilot 当作完整
结果。第一次完整 run 结束后先保留结果，再用独立 verify 进程重新推理、求解、
评分；只有全部一致才声称 exact replay。它仍不是独立科学复现样本。

## 判断是否仍在运行

```sh
pgrep -fl run_m3w_signed_utility_allocation.py
cat data/stage_cvpr2027_experiments/european_signed_utility_allocation_v1/heartbeat.json
tail -20 data/stage_cvpr2027_experiments/european_signed_utility_allocation_v1/readout.log
```

用真实 PID、日志和 heartbeat 的时间/查询编号共同判断；不能仅凭 lock 文件、
旧状态记录或一次观察超时就重启。健康运行可继续到12小时。每250个 query
以及每个完成组有恢复点，磁盘保留10GiB。若真实进程已结束且阶段未完成，
先保留失败日志和 checkpoint，再恢复同一阶段，不修改数据、预算或模型：

```sh
.venv-pytorch/bin/python scripts/run_m3w_signed_utility_allocation.py run --resume
```

如果是 verify 中断，重新执行 verify；它重新计算全部组，不覆盖首次 readout。
不要同时启动两个相同任务。进入 CREATE 前需要重新核对访问、调度器、现有
作业和目标环境；本说明不声称当前远程队列已经核实或实验已在远程运行。

## 公开轻量结果与完整私有复现

完整结果产生后，公开 ZIP 只含逐组汇总 JSON，不含原始行或权重：

```sh
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_signed_utility_allocation_v1 --verify
.venv-pytorch/bin/python scripts/report_m3w_signed_utility_allocation.py
```

第一条只检查 archive、manifest 和每组原始 JSON 字节；第二条在完整 replay
收据存在时重新汇总72组。它们不重新训练、推理或求解。完整私有复现还需要
registration 绑定的原始数据、特征及风险头 checkpoint，Git 中不提供这些。

```sh
.venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_signed_utility_allocation.py \
  tests/test_m3w_signed_utility_reduction.py \
  tests/test_m3w_signed_utility_reporting.py
```

最后一个文件包含依赖完整真实结果、replay 和报告的 artifact 检查，不能在
这些前提缺失时声称通过。数学反例和合成穷举测试只检验实现，不检验真实效果。

## 读结果时的规则

主要指标是同一当前 query 内切换数量严格相同的 completion-lower 效用差；
free-count 对照是次要结果，不能用它替换失败的主要结论。原定2% all/easy
positive-harm/reference 要求保留全部录制视图，收益不用于抵消正伤害分子。
空风险保持不可估，未知未来不补成安全零。oracle 只在标签侧诊断，不是模型。

12个 locality 已用于开发，三个风险头种子共享一个预测器种子，因此置信区间
不是独立新场景确认，也未校正长期自适应搜索。无论本轮结果如何，都不能把
source 拟合提升当作可部署、跨域世界模型或投稿准备完成。
