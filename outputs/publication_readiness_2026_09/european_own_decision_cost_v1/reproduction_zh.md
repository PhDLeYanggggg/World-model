# 固定选择与当前选择成本训练：复现说明

## 本轮训练什么

本轮是真实Torch成本模型训练，不是新轨迹预测器训练。两臂从同一已核验的
learned-reference权重出发，共享31,119个可训练参数、自洽输出变换、采样和预算。
区别只在查询级风险损失：固定旧选择，或根据当前预测成本重新求出的联合选择。
选择过程不接收未来标签或标签是否存在的信息；标签仅进入损失与评价。

这是已暴露SOURCE上的开发拟合，不是独立验证。观察8步、预测12步、rawstride12。
EuropeanSquares是斜视图像局部坐标、detector-silver，不是SDD t+50、米、秒、
人工gold、true3D或foundation。独立数据角色、Stage5C与SMC均不打开。

## 环境与资源

使用原生arm64 `.venv-pytorch/bin/python`，CPU4、interop1、worker0；入口拒绝
x86/Rosetta后再导入Torch。不探测Torch资源、不启用多进程DataLoader。
真实400-update试跑用41.121026秒，峰值9,816,473,600字节，两臂训练恢复和
18-query原生分配恢复一致。单一上下文估算正式拟合约8,400秒，不是完成时间。
本机48GB内存适合继续；没有因此新增CREATE任务或声称远程队列已验证。

## 训练、评价与恢复

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_own_decision_cost.py train
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_own_decision_cost.py eval
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_own_decision_cost.py verify
```

正式训练包含36个成本模型、208,320次主更新，以及两次完整预算重训核验的
14,336次更新。每条查询至少采样四轮，不因慢而减小数据或预算。
查询级监督分别计数非空且完整、含未知标签、空选择，空查询不冒充有效监督。
当前选择在训练中停止梯度；这不是对整数求解器求导或已经实现公开DFL定理。

私有目录：`data/stage_cvpr2027_experiments/european_own_decision_cost_v1/`。
训练每50步保存模型、优化器、采样器和RNG；分配评价每250查询保存进度。
心跳和日志记录PID，活跃分配至少每分钟更新一次。单phase上限12小时，预留
10GiB磁盘和1GiB本实验检查点限额。先检查真实PID和日志，不因观察超时重启。

实际中断后，在相同训练或评价命令后加 `--resume`。不变更协议、权重来源或
输入。过程锁防止重复运行；已完成phase不可覆盖。恢复核验排除实际计时值，
不排除模型、优化器、损失、采样或决策统计。限时求解若导致不同决策，必须
记录为实际复现失败。最终verify重推成本、重评分冻结动作，不重跑限时MILP。

## 公开报告复现

完整训练和评价需现有私有输入与上游权重，不是纯Git冷启动。完成后可仅用
Git中的标量报告和归档重建公开结果：

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/verify_m3w_own_decision_public_report.py
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  .venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_own_decision_cost.py tests/test_m3w_own_decision_evidence.py \
  tests/test_m3w_own_decision_public.py tests/test_m3w_refit_cost_allocation.py
```

报告重建保护在Python文件读取层阻止私有输入，不是操作系统隔离认证。
私有PNG仅用于输出和视觉检查；Git只收代码、配置、报告、轻量标量与科学SVG。
原始数据、特征、缓存、模型权重、视频图像、第三方PDF与环境均不提交。

## 怎样判断结果

主指标是own相对fixed完整completion-lower效用的配对区间，不用单个平均ADE
替换。还需超过冻结旧策略、超过floor且每条原始录像满足原定2%要求。
后验同数量裁剪只比较效用；空对照和无支持风险是不可估，不是安全零。
SOURCE bootstrap按地点配对，不把重叠窗口和重复context当独立样本。
三个成本种子共享forecaster43，不是三套预测器复现。缺少请求终点时FDE不可估，
不换更早终点，不编造未来内部点。无论拟合是否提升，本实验不自动授权部署。
