# 因果成本自洽投影：复现说明

## 实验范围

本轮没有训练新神经网络，也没有改变已有轨迹预测器。36组成本权重和七种
轨迹预测保持冻结；只将同一agent可用候选的预测参考量取均值，并限制收益不超过
共享参考量。伤害、easy伤害、候选可用性和真实标签不变。没有未来标签参与投影。

这仍是已暴露SOURCE上的拟合评价。观察8步、预测12步、rawstride12，不能和
历史SDD t+50混为同一任务。EuropeanSquares是斜视图像局部坐标、detector-silver，
不能宣称米、秒、人工gold、true3D或foundation。

## 本地运行与恢复

使用原生arm64 `.venv-pytorch/bin/python`，CPU4、interop1、worker0。
本轮试跑和旧完整评价均能放进本机48GB内存，不需要新增CREATE任务。
原CREATE只读访问收据可核验历史访问，不等于当前队列或GPU环境认证。

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_coherent_cost_projection.py pilot
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_coherent_cost_projection.py eval
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_coherent_cost_projection.py verify
```

这些计算命令需要已有私有数据、上游冻结权重及原始完整评价缓存，不是纯Git
冷启动训练复现。已完成的结果不可覆盖；没有训练phase。真实中断后，在相同
eval命令后加 `--resume`，读取同一定义、输入和已完成查询的检查点。
先检查PID和日志，不因一次观察超时而重复启动。过程锁阻止同一实验重复运行。

每250查询保存分配进度并写心跳，phase上限12小时，预留10GiB磁盘。
私有日志/检查点在 `data/stage_cvpr2027_experiments/european_coherent_cost_projection_v1/`。
完整核验重新推断成本并重评分已冻结动作，不是重跑全部限时MILP优化。

## 仅用公开标量重建报告

完整核验后运行：

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/verify_m3w_coherent_projection_public_report.py
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  .venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_coherent_cost_projection.py \
  tests/test_m3w_coherent_projection_controls.py \
  tests/test_m3w_coherent_projection_evidence.py \
  tests/test_m3w_refit_cost_allocation.py
```

公开报告不读取私有数据、特征或权重；私有PNG只用作输出和视觉检查。
Python文件读取保护不等于操作系统级隔离。原始论文PDF、原数据、权重、缓存、
大episodes及环境均不提交Git。

## 解读边界

预测矩自洽是本后处理的结构性质，不是校准或方法成功证明。参考MSE下降也
可能只是均值操作的数学结果。完整主指标和每条原始录像的2%要求均保留。
同切换数量对照只检验效用；空对照与没有easy参考支持是不可估，不是安全零。
bootstrap在重复context/成本种子平均后按地点配对重采样，不将重叠窗口当独立样本。
三个成本种子共享forecaster43，不是三套独立预测器训练。
缺少请求终点时FDE仍不可估，不替换更早终点，不编造缺失帧。
Stage5C与SMC始终关闭，独立数据不因SOURCE结果不好而打开。
