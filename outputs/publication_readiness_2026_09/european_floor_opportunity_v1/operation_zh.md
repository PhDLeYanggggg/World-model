# 保护基线增量诊断：运行与核验

## 本轮做什么

读取并校验两种已完成的 cross-moment 模型，不重新训练，也不选择模型。
把神经策略相对保护后阻尼基线的误差拆成：抓住的收益、切错的伤害、
错过的收益、回退到 CV 损失的收益，以及 CV 偶尔比阻尼更好的收益。
所有未来标签只用于事后误差统计，不能进入切换决策。

本轮是已打开开发场景上的新诊断，不是独立测试，也不重新认证历史 Stage37。
新回退规则没有相对于新基线校准风险，因此即使分数为正也不直接部署。
默认保留两个模型版本、三个种子、三个场景组和两种风险目标。

## 本地运行

在项目根目录使用原生 arm64 环境，单进程，CPU 计算线程 4，交互线程 1。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_floor_opportunity.py --mode batch --pilot
.venv-pytorch/bin/python scripts/run_m3w_european_floor_opportunity.py --mode batch --resume
.venv-pytorch/bin/python scripts/run_m3w_european_floor_opportunity.py --mode fitting --resume
.venv-pytorch/bin/python scripts/report_m3w_european_floor_opportunity.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_floor_opportunity.py tests/test_m3w_floor_protocol.py tests/test_m3w_floor_reporting.py
```

每组完成后存入私有目录；进度文件记录 PID、UTC、当前组及耗时。
中断后使用相同命令和 `--resume`。已完成组只校验并复用，不反复重算。
禁止修改已登记的分析脚本后继续覆盖旧结果；需要修复时记录修订和原因。
磁盘少于 10GiB 时停止并保留产物，不自动删除旧数据。

## 结果在哪里

- `results.md`：全部 36 组的误差分解，不只展示有利种子。
- `summary_metrics.json`：总体核查及分组摘要。
- `batch/`、`fitting/`：每组的指标、区间、安全检查和逐场景聚合表。
- `floor_rebase.svg`：相同神经决策、不同回退动作的配对结果。
- `completion_checks.json`：代码、上游证据、结果摘要的校验和。
- 私有 `data/stage_cvpr2027_experiments/european_floor_opportunity_v1/`：
  完整分组结果、过程日志、测试记录和只读 CREATE 查询回执，不上传 Git。

## 如何判断是否成功

诊断完成不等于神经世界模型成功。必须分别看原策略和离线改回退策略，
同时保留正误差 easy 子集与零 CV 误差样本，不能把后者从安全结论中删掉。
完整标签子集只用于敏感性分析，不代替主评价总体。
oracle 用未来真值选最优轨迹，是不可部署的收益上界，不是真实模型。

区间采用 3,000 次场景配对抽样；不同种子和重复场景不是新的独立样本。
本轮单位仍是图像像素，观察 8 步、预测 12 步、rawstride12。
不能改写为历史 t+50、秒级、米制、物理安全、人工金标准、true 3D 或 foundation。
Stage5C 与 SMC 继续关闭。需要相对强基线重新学习并独立校准后，才可讨论部署。
