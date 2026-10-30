# 因果位置估计对照：运行与结果核验

## 本轮实际做了什么

不是新一轮 JEPA、Transformer 或大模型训练，而是一个明确的基线机制对照：
保持过去4/8步估计的 OLS 速度不变，只把预测起点从最后观测位置改成过去
轨迹线性拟合所得的当前位置。模型预测只读取过去，未来标签只用于误差统计。

原始数据、划分、floor、旧物理预测与 producer43 身份是 cached_verified。
24组完整预测和误差重算、独立进程整轮 replay 是 fresh_run。误差几何分解
在主结果为负后提出，再在自己的数值计算前登记，因此是 post-hoc，不是新的
主要结果。独立校准、确认测试、新神经训练、部署均为 not_run。

任务是观察8步、预测12步、间隔12 raw frames，图像局部坐标、倾斜视角、
detector-silver 标签。不是 SDD raw-frame t+50，也不是秒、米、human gold、
true3D、foundation 或跨数据集成功。Stage5C 和 SMC 均未执行。

## 运行环境

项目根目录 `/Users/yangyue/Downloads/World` 使用原生 arm64 的
`.venv-pytorch/bin/python`，不要使用 Intel Conda/Rosetta。
入口在 import Torch 前检查架构。CPU线程4、interop1、workers0，不做资源
探测、不启动 DataLoader 子进程。实际位置估计由 NumPy 闭式公式计算，不能
把它说成 Torch 神经训练成功。本轮约6.11GB峰值内存，适合48GB本机，未提交
CREATE，也未重启或取消远程旧任务。

## 原始执行顺序

以下是本轮首次执行的顺序，不是要求覆盖已有完成结果：

```sh
.venv-pytorch/bin/python scripts/run_m3w_position_anchor.py register
# 提交科学代码、配置、协议、registration 后，才开始自己的实测。
.venv-pytorch/bin/python scripts/run_m3w_position_anchor.py pilot
.venv-pytorch/bin/python scripts/run_m3w_position_anchor.py run
.venv-pytorch/bin/python scripts/run_m3w_position_anchor.py verify
# 主实验完整重算为负之后才登记这个 post-hoc 诊断。
.venv-pytorch/bin/python scripts/diagnose_m3w_position_anchor.py register
# 提交诊断定义和 registration 后运行。
.venv-pytorch/bin/python scripts/diagnose_m3w_position_anchor.py run
.venv-pytorch/bin/python scripts/diagnose_m3w_position_anchor.py verify
.venv-pytorch/bin/python scripts/report_m3w_position_anchor.py
```

已完成的 run/pilot 不覆盖。首次完整 verify 必须重新计算24组，不只是检查
hash。现有 verify 的运行时收据也是 immutable，因此不要在已完成目录反复
执行同一 verify 并期望更新 PID/耗时；再次科学复算需保留第一次证据，使用
单独登记的运行记录，不删除已有科学结果。本轮已完成一次独立进程的全量
父实验 replay，以及一次诊断 replay，均精确一致。

## 现在可直接运行的公开汇总核验

```sh
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_position_anchor_v1 --verify
.venv-pytorch/bin/python scripts/report_m3w_position_anchor.py
.venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_position_anchor.py \
  tests/test_m3w_position_anchor_readout.py \
  tests/test_m3w_position_anchor_runner.py \
  tests/test_m3w_position_anchor_mechanism.py \
  tests/test_m3w_position_anchor_reporting.py
```

report 命令同时检查主实验及 mechanism 两个汇总包。通用 bundle 命令只接受
一级实验名，不使用带斜杠的子目录参数。这几条检查科学定义、完整 replay 收据和规范化汇总，并
重建报告。它们不重新计算私有逐行轨迹，不能写成 fresh 全数据实验。真实
重算还需要 registration 绑定的私有原始输入、历史缓存、floor 及旧预测。
公开 ZIP 只包含逐组汇总 JSON，没有轨迹、权重、图片或第三方视频。

## 若真实进程中断

先检查 PID、日志和 heartbeat，不能因一次超时就重复启动：

```sh
pgrep -fl 'run_m3w_position_anchor.py|diagnose_m3w_position_anchor.py'
cat data/stage_cvpr2027_experiments/european_position_anchor_v1/heartbeat.json
cat data/stage_cvpr2027_experiments/european_position_anchor_v1/mechanism/heartbeat.json
```

健康运行允许继续到12小时，保留10GiB磁盘余量。每个完成组有不可变恢复点。
只有确认进程结束且24组未完整完成，保留日志后恢复同一配置：

```sh
.venv-pytorch/bin/python scripts/run_m3w_position_anchor.py run --resume
.venv-pytorch/bin/python scripts/diagnose_m3w_position_anchor.py run --resume
```

两条是分别针对对应失败阶段的命令，不是同时启动。verify 若在运行时收据
产生前中断，可重算完整 verify；已完成 verify 不按恢复处理。

## 结果怎么读

主要 OLS8 fitted-minus-raw 的 known utility 为 -19.349820pp，completion-
lower 为 -21.078204pp，区间分别为[-37.826305,-9.344567]、
[-41.400334,-10.049531]。OLS4 也为负。4/7拟合检查通过，主要科学条件未过。
所有4个直接 gated 候选均为0/148原始 recording/context 风险视图通过。
这里 gated 只指原有因果 eligibility，不代表学到的安全切换器。

12个 locality 都已暴露用于开发，3000次 bootstrap 是条件性名义区间，不是
独立新场景确认。producer43 不是3个新预测器种子。149510是完整私有 ID-set
检查确认的唯一行数，不能通过重复 context 行数相加获得独立样本量。
缺失完整 endpoint 的 FDE 保持 not_estimable，不可将有标签录像的平均值
替代完整队列指标。hard/easy 池化结果、逐录像尾部、pair-distance proxy 都
是各自覆盖范围内的补充结果，不替换主要结果或原定2%正伤害预算。

post-hoc 恒等式发现 cross term 在12个来源均为正，说明本轮位置改动平均
强化已有预测误差。但不能据此认定是噪声、转弯或检测误差中的哪一种原因。
保留原预测和原部署，不在这轮负结果上继续调窗口或放宽风险规则。
