# 本轮结果与操作说明

## 本轮实际完成

本轮回答一个具体问题：如果简单阻尼运动预测也获得相同的收益学习、风险学习和
回退保护，神经轨迹预测还能否证明额外价值？不是再扩大模型，也没有打开独立测试。

已新训练 45 个阻尼候选控制头，其中 18 个岭回归、27 个真实 PyTorch 小网络，
合计 54,000 次神经更新。神经轨迹候选已有的 45 个控制头经过校验后复用，没有
冒充本轮重新训练。两个候选的决策都重新执行，48 个固定策略视图全部完成。

## 结论是什么

目前结果不支持神经轨迹预测优于简单运动候选。24 组对应的逐 agent 对照中，
神经候选相对受保护阻尼的改善均为负，约 -1.83% 到 -0.22%，各自的条件性
地点 bootstrap 区间都在零以下。hard 子集和联合决策样本中的对照也没有反转。
这些是同一批源场景和共享模型上的结果，不是 24 次独立验证。

以前一轮重点报告的 easy 神经风险头为例，三个种子的阻尼候选相对 CV 改善
为 1.8108%、1.9388%、1.8975%；神经轨迹候选为 0.2382%、0.1663%、0.4310%。
阻尼这组在所有已观察地点的 easy 上均未退化，也未伤害四个 CV 零误差样本。
这说明风险头可能有用，但不能据此把神经风险学习写成神经世界动力学成功。

负结果仍须保留：阻尼的另一组 easy 岭回归策略在三个种子的最差地点都超过
2% easy 退化。相同介入数下的联合决策优势没有建立。联合样本中没有 CV 零
误差案例，不能以“零受损”证明稀有事件安全。完整表见 [results.md](results.md)。

当前不更换部署，不宣布论文候选达标。主任务是观察 8 步、预测 12 步、原始帧
间隔 12、图像像素坐标，不能混入历史 t50 数字，也不能说成秒、米、true 3D、
foundation 或人工 gold。后续审计发现历史 Stage37 存在来源与选择暴露问题，
旧高分只能作为探索性历史，不能替代当前证据。Stage5C、SMC 均未执行。

## 如何运行和恢复

工作目录：`/Users/yangyue/Downloads/World`。
必须使用原生 arm64 `.venv-pytorch/bin/python`，不能用 x86_64 Conda。
计算线程 4、inter-op 1、DataLoader workers 0；入口在加载 Torch 前检查架构。
首次训练的固定命令为：

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_protected_motion.py --prepare --pilot
.venv-pytorch/bin/python scripts/run_m3w_european_protected_motion.py --train --resume --evaluate
```

100 次真实更新试跑包含在正式预算内。中断时使用第二条命令恢复模型、优化器
和随机状态，不重复启动现有进程。`events.jsonl`、`heartbeat.json` 记录 PID、
时间和当前阶段；慢但仍在前进不是卡死。身份不一致须诊断，不能覆盖检查点。

完整复核与生成报告：

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_protected_motion.py --verify --replay
.venv-pytorch/bin/python scripts/report_m3w_european_protected_motion.py
.venv-pytorch/bin/python scripts/plot_m3w_european_protected_motion.py
```

复核不是重新训练。当前完整指标一致，90 个头各 4,096 行重新推理完全一致。
九组训练抽样各在六个神经头之间一致；144 份决策记录的场景、帧和实际介入数
通过核对。同风险预算不等于不同候选之间同介入数，报告明确区分这两种对照。

公共目录：`outputs/publication_readiness_2026_09/european_protected_motion_v1/`。
私有训练目录：`data/stage_cvpr2027_experiments/european_protected_motion_v1/`。
公共 `summary_metrics.json` 保存精简指标；详细 `analysis.json` 在本地生成，
本次不将约 17.67 MB 的详细展开表提交到 Git。没有本地私有输入的机器只能阅读
公共结果，不能因此声称完成了训练复现。

本轮所有训练与核验进程已结束。本地资源足够，未提交 CREATE 作业；既有
simulation 项目限制不变，不修改其目录或运行任务。未来需要 HPC 时先确认
可用目录、环境和调度限制，再提交可恢复任务，不凭旧记录宣称已经连接或开跑。

## 后续重点

先拆解神经候选为什么在保护下丢失收益：预测本身的增益/伤害、风险估计偏差，
还是生成训练标签与最终预测的模型训练规模不一致。诊断只使用已开放的源场景。
再冻结一次单因素修复实验，保持风险上限，不使用独立确认集调阈值。

常规审计、运行、负结果记录与安全 Git 更新由项目流程自行完成，不再要求你
逐项审计。这不等于跳过审计，也不取消正式投稿的最终确认或科研诚信边界。
