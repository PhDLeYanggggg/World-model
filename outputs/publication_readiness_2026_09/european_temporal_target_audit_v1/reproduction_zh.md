# 时间目标诊断复现说明

## 实验做什么

本实验用已经冻结的因果树分区，在 TRAIN 上拟合逐步误差均值探针。
输入仍是原有历史特征和候选预测分歧，没有未来标签或未来有效性掩码。
未来坐标只用于监督目标和离线评价。对照是全局逐步均值，以及树叶内整段均值。
它不是新神经网络训练，不改变部署模型。

重点区别：逐步正伤害平均值通常大于整段平均误差的正增量，两者不能直接替换。
原 B/H/R/ER/EH 风险目标、2% 预算、切换动作和未知标签补全规则不变。

## 注册与修正

初始注册提交 `c7759278`，原试跑保存为 `pilot.json`。
修正注册提交 `d046c265`，记录在 `amendment.md` 和 `registration_amended.json`。
修复了报告中 ER=0、EH>0 的 easy 伤害可能漏计的问题，未修改模型或预算。
原注册和试跑不删除。12 项针对性测试通过；修正试跑约 32.75 秒、峰值 6.69 GB。

## 运行

工作目录 `/Users/yangyue/Downloads/World`，使用 arm64 环境，4 个计算线程，
DataLoader worker 为 0。不要使用 x86 Conda/Rosetta。

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_temporal_target_audit.py run
```

已经完成的实验不可覆盖。若中断，先查实际进程是否仍活着，再用相同命令加
`--resume`。已有组从原始资产确定性重算并逐项核对，未完成组继续；不是删除重跑。
只保存每组轻量报告与哈希，无新数值缓存或模型权重。心跳和进度锁在
`data/stage_cvpr2027_experiments/european_temporal_target_audit_v1/`。

## 复核

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_temporal_target_audit.py verify
env PYTHONDONTWRITEBYTECODE=1 \
  .venv-pytorch/bin/python scripts/verify_m3w_temporal_target_audit.py
```

第一条完整重新拟合诊断均值并重放推理，不是新神经训练；第二条独立核对
报告标量、原动作哈希与 locality bootstrap。不能把只读报告称为重训。
3,000 次 bootstrap 的单位是 locality，不是重叠窗口或 72 个独立场景。
所有区间是已暴露开发数据上的名义区间，不是独立风险保证。

## 本轮实际完成

72/72 组完成，完整运行 279.13 秒，峰值内存 9.719 GB。
每组在运行内完成确定性重新拟合和推理重放；1,656 项运行检查通过。
独立报告复核完成 3,696 项标量与 bootstrap 检查。上面的 `verify` 命令
另外提供事后全量重放入口，本轮没有再重复执行这次额外全量重放。
12 项针对性测试已经通过；本轮没有执行全部历史测试套件。

完整结果由 `complete.json`、`summary.json`、`verification.json` 共同证明。
对树叶内整段均值，逐步 signed-error MSE 改善 0.094342，名义区间
[-0.158824, -0.038614]；但旧策略实际选择样本上的区间包含零。
因此通过的是“可以继续测试时序辅助监督”的诊断门槛，
不是轨迹模型、风险控制、部署或投稿质量门槛。

报告生成可重复执行：

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  .venv-pytorch/bin/python scripts/report_m3w_temporal_target_audit.py
```

[完整解释](conclusions.md)、[结果表](results.md)、
[统计局限](statistical_interpretation.md)保留了负结果和后续实验约束。
CREATE 本轮两次只读连接超时，无法观察远程作业，并不证明作业失败；
不要因此重启任务。本地实验不依赖远程读写，没有新增大缓存或权重。

保持 image-local、detector-silver、obs8/pred12、rawstride12 表述，
不声明米、秒、human gold、true 3D、foundation。Stage5C/SMC 关闭。
