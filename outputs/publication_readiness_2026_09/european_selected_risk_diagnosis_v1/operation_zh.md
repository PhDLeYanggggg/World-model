# 冻结策略诊断复现说明

## 这次做了什么

本轮重新计算已冻结策略的风险残差、训练特征覆盖、切换带来的收益和伤害。
模型及动作来自经过哈希核验的已有结果，不是重新训练。只使用已经开放的
12 个开发场景，独立选择、校准、确认数据没有打开。

## 运行

在项目根目录使用本地 arm64 环境，计算线程4、互操作线程1、worker0。

```sh
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m pytest -q tests/test_m3w_selected_risk_diagnosis.py
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/diagnose_m3w_selected_risk.py --resume
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/diagnose_m3w_selected_risk.py --replay
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/report_m3w_selected_risk_diagnosis.py
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/verify_m3w_selected_risk_diagnosis.py
```

`--resume` 核查已完成分组并接续；`--replay` 真正重新推理和计算全部分组，
要求结果完全一致。每组都会更新本地 heartbeat，记录 PID。剩余磁盘不足
10GiB 时保留已完成记录并退出，不删数据、不降低预留标准。

私有运行记录位于 `data/stage_cvpr2027_experiments/european_selected_risk_diagnosis_v1/`。
公开报告位于同名 `outputs/publication_readiness_2026_09/` 子目录。
不要把私有逐组记录、数据缓存、模型权重或环境提交到 GitHub。

## 怎样读结果

正的 optimism 表示风险低估，单位是训练集成本归一化后的值，不是百分数。
平均简单样本误差保持，不等于每个切换群都符合风险要求。
空切换的条件风险仍是 undefined，不能替换成0；总基线误差分母的收益分账
也不能冒充原来的2%条件风险证书。

12 个场景中反复出现的窗口不是独立样本；先在场景内平均，再做场景 bootstrap。
部分切片无样本时保留缺失计数，不为得到 CI 删除场景。

首次计算用时306.87单调时钟秒，峰值内存11.65GB。复算日志曾出现 UTC
时间间隔明显大于单调时钟增量的情况，随后继续完成；没有据此判为卡死或
重新启动。日志间隔本身不能证明具体原因。CREATE 本轮仅只读查队列，没有
新提交或修改远程作业。

这些是开发诊断，不是部署、独立校准或投稿准备完成。raw-frame 不代表秒，
image-local 坐标不代表米，detector silver 不是人工 gold。Stage5C/SMC 均未启用。
