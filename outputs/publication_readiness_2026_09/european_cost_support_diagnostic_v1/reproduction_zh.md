# 冻结成本头诊断：复现说明

## 本轮做什么

重读已经完成的 72 个成本预测头，验证权重、输入、预测、动作和原有评价值，
再计算训练支持与误差分解。不训练新头，不调整切换阈值，不打开独立选择、
风险校准或最终确认数据。所有 12 个 locality 均为已经暴露的开发样本。

主要区分三个问题：

1. 每棵树训练损失下降后，整个森林的训练误差是否也下降。
2. 训练改善是否保持到同一 locality 中整段录像留出的验证集。
3. 误差是否集中在少量训练伤害事件、质量特征支持不足或大的乘性变化中。

分层结果只能定位关联，不能单独证明因果。未知未来标签仍然保留为未知。
风险预算仍为选中样本的正向 easy 伤害之和除以对应参考误差之和，预算 2%；
不能改写成全体 easy 样本的净退化。

## 环境与产物

在 `/Users/yangyue/Downloads/World` 使用原生 arm64 `.venv-pytorch/bin/python`，
4 个计算线程、0 个 DataLoader worker。CREATE 仅用于读取项目自己的冻结权重，
不提交新调度任务、不修改 simulation 项目。SSH 配置及主机密钥校验保持原样。

代码：`scripts/run_m3w_cost_support_diagnostic.py`。
参数：`configs/m3w_european_cost_support_diagnostic_v1.json`。
注册：本目录 `registration.json` 与 `protocol.md`，运行前提交为 `5ce4f685`。
心跳与事件：`data/stage_cvpr2027_experiments/european_cost_support_diagnostic_v1/`。
逐头轻量结果：本目录 `groups/`，不包含逐行轨迹或模型权重。

## 执行与恢复

首次完整运行的命令如下；已有完成结果时不要覆盖后重新训练：

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_cost_support_diagnostic.py pilot
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_cost_support_diagnostic.py run
```

如中断且没有 `complete.json`，先确认原进程已经退出，再使用同一配置与注册：

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_cost_support_diagnostic.py run --resume
```

恢复会重算并精确核对已有逐头报告，不覆盖不同内容。完成后重新核验无需训练：

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_cost_support_diagnostic.py verify
env PYTHONDONTWRITEBYTECODE=1 \
  .venv-pytorch/bin/python scripts/verify_m3w_cost_support_diagnostic.py
env PYTHONDONTWRITEBYTECODE=1 \
  .venv-pytorch/bin/python scripts/summarize_m3w_cost_support_diagnostic.py
```

第一条重放真实权重和输入；第二条使用独立标量运算与 locality bootstrap 核对报告；
第三条整理失败切片。单独运行后两条不等同于重新训练或重新读取全部权重。

## 验证边界

真实试跑已完成：48.27 秒，峰值内存 6.47 GB，92 项独立标量检查。
全量 72 个头已完成：478.13 秒，峰值内存 9.22 GB。25 项相关测试通过，
运行器有 6,624 项独立策略标量检查，最终核验有 13,390 项标量与 bootstrap 检查。
训练误差全部下降，但 43/72 个头的最终验证误差上升；本轮没有提升部署状态。
逐树损失重构容差在运行前固定为 `1e-6`，实际最大残差为 `1.23e-15`；
父实验分数与误差分解仍为 `1e-9`。
最终以 `complete.json`、`verification.json` 和 `summary.json` 为运行证据，
不能只凭心跳或进程退出宣布研究成功。bootstrap 重采样单位是 locality，
不是高度重叠的窗口；区间为暴露开发集上的名义区间，不是独立泛化保证。

保持 detector-silver、image-local、obs8/pred12、rawstride12 的表述。
本轮不证明秒、米、物理安全、true 3D、foundation 或投稿就绪；Stage5C/SMC 均关闭。
