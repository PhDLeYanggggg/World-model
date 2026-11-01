# 完整查询覆盖成本修复：复现说明

## 范围

本轮训练的是连续成本网络，不是新的轨迹预测器，也不是生成式世界模型。
比较从相同旧权重继续训练的两组：固定参考量与学习参考量。
两组使用相同因果特征、旧预测、归一化、查询顺序和完整预算。
学习组多262个活动参数，不应称为完全活动容量匹配。

所有结果均为已暴露欧洲来源数据上的拟合评价，不是独立验证。
观察8步、预测12步、raw-frame stride12，不能混入旧SDD t+50结果。
图像局部坐标、检测器silver标签，不是米、秒、人工gold、true3D或foundation。

## 环境与恢复

使用本机arm64 `.venv-pytorch/bin/python`，CPU线程4、interop1、worker0。
不要用x86 Conda/Rosetta环境。入口在导入Torch之前检查架构。
脚本每100步保存模型、优化器、随机状态、循环采样器和轨迹；每50步写心跳。
每个phase最多12小时，预留10GiB磁盘；健康慢任务无需缩减目标。

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_reference_cost_refit.py pilot
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_reference_cost_refit.py train
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_reference_cost_refit_readout.py eval
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_reference_cost_refit_readout.py verify
```

这些训练命令需要原始私有素材、已核验旧权重与冻结上游结果，不是纯Git冷启动复现。
已完成的pilot、training freeze及readout不可覆盖；不要重复启动。
真实中断时在相同phase后加 `--resume`，不能另起不同预算冒充恢复。
过程锁避免重复训练。首先检查PID和私有心跳，不因观察失败而重启。

原评价入口在计时变量处存在Python局部作用域错误，首次评价在runtime_ready前
失败，尚未产生任何新策略组。原注册、科学文件与36组权重均保留不变。
修复入口只在内存中的原评价语法树上重命名1处局部绑定及2处引用；反向改名
必须逐节点恢复原语法树。`execution_erratum.json`记录前瞻冻结的补丁和哈希。
算法、预算、loss、标签、阈值、原注册和模型未改动；不是重新选择实验。

私有日志和checkpoint在 `data/stage_cvpr2027_experiments/european_reference_cost_refit_v1/`。
公开注册、每模型收据、训练冻结、72组标量压缩包和最终摘要在本说明同目录。
私有数据、特征、缓存、权重、图像和环境均不提交Git。

## 仅用公开轻量证据重建报告

完整数值核验完成后运行：

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/report_m3w_reference_cost_refit.py
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/verify_m3w_reference_public_report.py
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  .venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_reference_cost_refit.py \
  tests/test_m3w_refit_cost_allocation.py \
  tests/test_m3w_reference_cost_evidence.py \
  tests/test_m3w_reference_runtime_erratum.py
```

报告从公开标量收据重建，不读私有数据或权重；私有PNG仅用于图形检查。
另行的完整模型推理/冻结动作重评分不是重新运行全部限时MILP优化。
样本中缺少请求终点时FDE仍为not_estimable，不能换成更早终点。

## 结论边界

每条原始录像都需满足原定2%风险要求。无easy参考支持是不可估，不是安全零。
同切换数量对照只检验效用，不能替代完整策略安全检查。
bootstrap按来源地点成对重采样，重复窗口及不同成本种子不当独立样本。
三个成本种子共用forecaster43，不是三套完整预测器训练。
训练loss下降、参考量MSE下降或label-aware oracle支持均不等于可部署成功。
Stage5C与SMC始终关闭，独立DroneCrowd标签不因本轮SOURCE结果而打开。
