# 固定输出幅度实验复现

## 实验含义

这轮不是再次训练成本头，而是用冻结的轨迹和成本估计器，比较原始输出与
1/2、1/4、1/8、1/16 幅度的实际输出。物理和神经候选使用相同规则。
每个设置同时保留原选择集合和允许重新选择两个对照。原始幅度必须精确
复现此前策略；零幅度只用于工程测试，不能当成安全模型。

推理只接受原始因果特征、完整预测轨迹和固定系数。未来目标与 mask 只进入
独立评分函数。评分重新计算实际新轨迹误差，不能直接缩放原误差。变换后的
成本是凸损失代理界，不代表原始估计已校准，更不是物理安全证书。

## 运行

在仓库根目录使用原生 arm64 环境，CPU 4 线程，worker 0。

```bash
.venv-pytorch/bin/python scripts/diagnose_m3w_forecast_amplitude.py register
.venv-pytorch/bin/python scripts/diagnose_m3w_forecast_amplitude.py run
.venv-pytorch/bin/python scripts/diagnose_m3w_forecast_amplitude.py run --resume
.venv-pytorch/bin/python scripts/diagnose_m3w_forecast_amplitude.py verify
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_forecast_amplitude_v1
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_forecast_amplitude_v1 --verify
.venv-pytorch/bin/python scripts/report_m3w_forecast_amplitude.py
.venv-pytorch/bin/python scripts/plot_m3w_forecast_amplitude.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_forecast_amplitude.py tests/test_m3w_forecast_amplitude_reporting.py tests/test_m3w_cost_bank_factorization.py tests/test_m3w_multiaction_cost_readout.py tests/test_m3w_metric_bundle.py
```

注册和绑定文件需先提交。首次执行使用 `run`，中断才用 `--resume`。
已经完成使用 `verify`，不能覆盖结果。运行保留不可变分组、私有 heartbeat、
进程锁、10 GiB 空间下限和 12 小时恢复边界。公开克隆不含原数据和模型，
命令不能在缺少私有资产时假装复现。

## 本轮记录

完整首次执行 PID 20388，退出 0，575.886368 秒，峰值内存
7,060,619,264 字节。独立重放以 `verification.json` 和 `evidence.json`
为准。两类估计器、六组上下文、三个成本头种子的 144 组、720 个幅度视图
全部保留；这些重复视图不是新的独立训练数据。

独立重放 PID 21457 已退出 0，574.746712 秒，所有分组及汇总完全一致。
720 次凸性数值核查通过，最大舍入超量为 6.82e-14 以下，不是学习风险保证。
最终针对性测试 28 项通过，1.65 秒；未重跑整套历史测试。

完整逐录像安全门未通过。不能只报告已观测误差下降、删除未知结局，或者
把空 easy 风险记成 0。拟合区间不是独立泛化证据，t+50 历史任务也不是
本轮 obs8/pred12/stride12 主任务。未打开独立评价或 DroneCrowd，未更改
部署，未执行 Stage5C，未启用 SMC。没有 metric、秒级或 foundation 声明。
