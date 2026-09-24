# 风险预算机制对照：运行与恢复

## 实验边界

本轮不是新的神经网络训练。复用已核验的 36 个风险估计头、因果特征和预测，
固定比较负风险信用与预算分母两项机制。四个 SDD 来源场景均已用于方法开发，
不得称为独立确认。观察 8 步、预测 12 步，annotation stride 12，像素坐标；
不混入历史 t+50 指标，不报告未经验证的秒或米。

原始数据、缓存、模型权重和 PNG 预览仅存本地。Git 只提交代码、配置、注册文档、
汇总指标与矢量图。DroneCrowd 不打开，Stage5C/SMC 不执行。

## 命令

在 `/Users/yangyue/Downloads/World` 使用原生 arm64 环境：

```bash
.venv-pytorch/bin/python -m pytest tests/test_m3w_causal_motion_support.py tests/test_m3w_risk_subsidy_controls.py tests/test_m3w_risk_subsidy_replay.py tests/test_m3w_net_easy_guard.py tests/test_m3w_net_easy_risk.py tests/test_m3w_native_scene_context.py tests/test_m3w_native_scene_alignment.py -q
.venv-pytorch/bin/python scripts/diagnose_m3w_zero_reference_support.py
.venv-pytorch/bin/python scripts/run_m3w_risk_subsidy.py --phase preflight
.venv-pytorch/bin/python scripts/run_m3w_risk_subsidy.py --phase decide --limit-chunks 1
.venv-pytorch/bin/python scripts/run_m3w_risk_subsidy.py --phase decide --resume
.venv-pytorch/bin/python scripts/run_m3w_risk_subsidy.py --phase evaluate
.venv-pytorch/bin/python scripts/run_m3w_risk_subsidy.py --phase decide --verify
.venv-pytorch/bin/python scripts/run_m3w_risk_subsidy.py --phase evaluate --verify
.venv-pytorch/bin/python scripts/verify_m3w_risk_subsidy.py
```

第一批试运行属于完整预算的一部分，不代替全量计算。每个批次原子保存结果与
来源 hash；恢复时必须使用 `--resume`，不会重新选择阈值。`--verify` 必须读取
已经存在的原始结果，不能把首次计算写成复现。

## 监控与证据

CPU 线程 4、interop 1、workers 0，不做资源探测。记录 PID、事件和心跳：

- `data/stage_cvpr2027_experiments/risk_subsidy_v1/heartbeat.json`
- `data/stage_cvpr2027_experiments/risk_subsidy_v1/events.jsonl`
- `data/stage_cvpr2027_experiments/risk_subsidy_v1/identity.json`
- `data/stage_cvpr2027_experiments/risk_subsidy_v1/decisions_complete.json`

慢不等于卡死，应核实同一 PID、CPU 时间、已完成批次和日志再决定处置。
不得仅因短暂没有输出就另启重复进程。长任务允许持续运行，批次恢复和单写锁
保留已有成果。只有全部 188,388 个 query/action/seed 实例完成后才进入读出。

新的决策全部冻结后才能读取评价标签。结果包括所有 33 个 action/policy 组合，
以及 easy 退化、零误差伤害、未知标签、最差场景、求解失败和匹配失败。
求解失败后的空集合不能伪装成正常零介入最优解，也不能视为零风险证明。
最多 10 个可介入 agent 的查询按原不等式穷举；更大查询中的非平凡数值解，
即便求解器报告最优，也单独标为原式最优性未验证，不能与失败空回退混淆。

运行完成、指标重放一致、独立算术检查均只证明相应计算环节；不意味着已建立
独立泛化、真实物理安全或论文录用保证。
