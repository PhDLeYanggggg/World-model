# 受保护简单运动对照：运行与核验

## 本轮回答什么

给简单运动公式和 Transformer 配上同样的增益/伤害学习与保护规则，
检验收益是否真的需要神经轨迹预测。实验已提前固定六种简单运动公式，
不是看到 DUT 哪个公式较好后只报告那一种。

四个 SDD 场景已经用于开发。即使逐场景排除训练、三种子和 bootstrap
均完成，也不能把它们改称独立确认场景。DUT 不再次评估，DroneCrowd
继续封闭。没有新部署、Stage5C 或 SMC。

## 环境与训练

工作目录为 `/Users/yangyue/Downloads/World`。使用原生 arm64 环境，
CPU4、interop1、workers0。不使用默认 x86 Conda。无需 GPU 或 CREATE
完成本轮小型控制头实验；这不代表后续大规模预测器训练也应留在本机。

```bash
.venv-pytorch/bin/python scripts/run_m3w_protected_motion_controls.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_protected_motion_controls.py --resume
```

共168个模型/折/种子单元：72个新神经控制头、84个新森林控制头、12个
既有 Transformer 控制头。后12个要核对输入、标签、预处理、样本支持、
预算、检查点哈希后才能复用。它们是 `cached_verified`，不是本轮新训练。

每500个神经更新、每16棵树保存原子检查点。日志及当前PID在
`data/stage_cvpr2027_experiments/protected_motion_controls_v1/` 下。
不要同时启动两个训练进程；锁文件会拒绝重叠运行。中断后保留目录，
使用同一 `--resume` 命令。不得更改冻结配置或代码来继续旧检查点。

## 训练完成后

```bash
.venv-pytorch/bin/python scripts/run_m3w_protected_motion_controls.py --evaluate
.venv-pytorch/bin/python scripts/run_m3w_protected_motion_controls.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_protected_motion_controls.py
.venv-pytorch/bin/python scripts/summarize_m3w_protected_motion_controls.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_protected_motion_controls.py tests/test_m3w_protected_motion_verification.py tests/test_m3w_bounded_cost_head.py tests/test_m3w_external_cost_bank.py
```

评估器要求完整矩阵完成。先固定并存储所有外层推理决策，再读取未来标签
计算误差。`--verify` 在新进程重放决策和汇总；独立算术脚本另算 ADE/FDE、
缺标签增益界限、选择数量和采样预算。它们不是另一名研究者独立复现。

主指标是四场景相对 CV 的 ADE 改善率等权均值，采用可用未来点。
完整轨迹、easy、hard、零CV误差、尾部及缺失标签另列。三种子误差平均
不等于对预测轨迹做集成。Bootstrap 按物理场景抽样，不能按重叠窗口
抽样来制造过窄区间。相同切换数量对照是离线分析，不是部署规则。

## 数据与结论边界

结果为像素坐标、原始标注步长，不能写成米或秒。固定观察8步、预测12步，
并未把 raw-frame t+50 改写为50秒。Scene/goal、多模态与联合交互贡献不由
本对照证明。EqMotion 的配对排除生产链尚未纳入此对照，必须明确缺项。

只提交代码、配置、汇总报告和轻量指标。不提交模型权重、逐行分数、原始
数据或任何缓存。更新 README 时保留负结果，不因为简单公式较强而隐藏它。
