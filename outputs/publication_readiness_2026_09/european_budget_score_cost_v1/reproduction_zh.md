# 原始风险预算分数监督：复现说明

## 研究问题与真实工作量

上一轮只替换神经轨迹候选，没有修好完整选择策略。本轮固定原有候选、数据、
特征和部署规则，只修改训练目标：保留完整五个连续成本量与排序损失，再监督
两个实际风险预算分数 `H-0.02R` 和 `H_easy-0.02R_easy`。

比较原损失、对称分数误差、风险低估加权4倍的分数误差。三个版本都保留原成本
回归，避免只学一个分数而让分子、分母失去约束。这不是硬分类，也不是重新选
测试阈值。方向损失是待检验方法，不是已经成立的安全保证。

完整矩阵为6个上下文、3个种子、3套候选库、3种目标，共162个真实成本头，
每个2,000步，共324,000次主要更新。另有首个完整保守新库模型2,000步复训，
以及420次真实试跑更新。54个原损失对照必须精确复现旧训练状态。不要把它说成
162个新轨迹预测器；上游轨迹模型和候选缓存沿用上一轮冻结资产。

## 环境

在项目根目录使用原生 arm64 `.venv-pytorch/bin/python`。不使用 x86_64 Conda
或 Rosetta。入口在导入 Torch 前检查架构，计算线程4、interop线程1、
DataLoader workers0，无多进程数据加载或 Torch 资源探测。保持检查点和心跳。

真实九组恢复试跑 PID77422 已完成：14.97秒、峰值9.78GB；全部10+10步与直接
20步精确一致，三个旧实现对照也一致。完整主要优化投影约808秒，不含数据准备、
完整复训和推理。最终实际时间与内存以 `evidence.json` 为准，不把投影说成实测。
本轮本机足够，无需新增或取消 CREATE 任务。

## 入口与顺序

配置：`configs/m3w_budget_score_cost_v1.json`。大数组、检查点和日志保存在
`data/stage_cvpr2027_experiments/european_budget_score_cost_v1/`，不进 Git。
公开目录保存注册、轻量收据、完整训练曲线、报告和聚合指标包。

```bash
.venv-pytorch/bin/python scripts/run_m3w_budget_score_cost.py register
.venv-pytorch/bin/python scripts/run_m3w_budget_score_cost.py pilot --resume
.venv-pytorch/bin/python scripts/run_m3w_budget_score_cost.py train --resume
.venv-pytorch/bin/python scripts/evaluate_m3w_budget_score_cost.py register
.venv-pytorch/bin/python scripts/evaluate_m3w_budget_score_cost.py run --resume
.venv-pytorch/bin/python scripts/evaluate_m3w_budget_score_cost.py verify
.venv-pytorch/bin/python scripts/report_m3w_budget_score_cost.py report
.venv-pytorch/bin/python scripts/report_m3w_budget_score_cost.py figures
.venv-pytorch/bin/python scripts/report_m3w_budget_score_cost.py bundle
```

代码、注册和训练冻结收据按顺序先提交，再进入依赖它的执行阶段。完成的实验
不能覆盖。中断后使用同一版本、同一检查点 `--resume`；已完成头只核验并跳过，
不能把跳过称为本轮新训练。初次完整 `run` 后使用 `verify`，不会改模型或阈值。

每100步原子保存，每50步写心跳，记录 PID、时间和梯度；有进程锁防止重叠执行。
12小时边界、10GiB剩余空间和512MiB私有检查点保护。日志在 `train.stdout.log`、
`readout.stdout.log`、`verification.stdout.log`，状态在 `heartbeat.json`。
观察工具超时不等于原进程结束，应先核实原 PID 和日志，不重复启动正常任务。

## 结果来源和统计单位

- `fresh_run`：真实成本训练、完整策略推理、另进程完整重跑。
- `cached_verified`：旧原始输入、冻结轨迹模型/缓存、分组与归一化，逐项哈希核验。
- `not_run`：控制器持出的89个录像结果、独立选择/校准/确认、DroneCrowd结果、场景联合决策。

149,510是去重查询身份数，不是独立统计样本数；相邻窗口和重复上下文/种子/
目标不能当作独立人或独立场景。主区间按12个locality重采样3,000次，先平均
上下文和种子。SOURCE历史上用于开发，即使完整重跑一致，也不会变成独立测试。

先看完整风险，再看平均准确率。所有必需录像都要满足原定2% all/easy要求；
平均 easy 净收益不能抵销受损样本的正伤害风险。同切换数量只衡量效用，空对照
必须“不可估”，不是安全零。缺失未来标签不能填成正确结果；全名单 FDE 若缺
必需终点支持，就保持不可估。不要删除负场景来获得显著区间。

## Loss 的解释与禁止项

曲线来自所有162个真实训练日志，不是验证集 loss。不同目标定义不同，不能以
损失数值大小直接排列模型优劣。必须再看固定主对照、神经真实使用增益和逐录像
风险。模板中的 gate 只是本轮拟合推进检查，即使通过也不自动授权部署。

SOURCE是 detector-silver 包围框中点 pixel，观察8步预测12步、stride12 raw
frames。不是历史SDD t50，不是秒、米、人工gold、已验证实时感知、true3D或
foundation证据。未来终点只用于监督/评估；没有 central velocity 或测试终点
目标。Stage5C未执行，SMC未启用。
