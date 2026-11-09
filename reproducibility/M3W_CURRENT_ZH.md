# M3W 当前论文与关键实验：中文复现入口

本文档对应 v130 研究稿。实验结果分三类：开发录像上的实测、使用未来标签的
事后上界、独立确认。前两类都不能写成第三类。DroneCrowd 的 15 个校准来源组
没有得到满足既定规则的策略，30 个独立确认来源组不用于以下命令。

## v130：广覆盖策略与风险误判

```sh
.venv-pytorch/bin/python scripts/run_m3w_cv_pooled_coverage.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_pooled_support.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_pooled_support_funnel.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_pooled_coverage_capacity.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_uncapped_dual_certificate.py
.venv-pytorch/bin/python scripts/build_m3w_cv_pooled_coverage_replay.py
```

第一条使用本机冻结模型和预测复核选择；第二到第五条是标签可见的诊断，不能
用来选正式策略。应得到 244 个无上限预测改动、校准 0/9，以及 53/72 个事后
双基线可行视图。容量求解逐录像保存，恢复前核对运行目录的
`capacity_heartbeat.json`、PID 及绑定哈希。最后一条生成只含匿名录像统计的
本地 ZIP；解压后在 NumPy 环境运行 `replay.py` 可复核标量表，不需原始数据
或 Torch，也不复现训练、邻居几何和求解器。

## v129：固定对象的候选轨迹与覆盖率下界

在本机冻结的拟合来源数据、风险模型和来源隔离预测可用时运行。第一条命令
加载两组模型检查点、验证预测并复算留出分数，不更新模型。另两条命令只做
事后容量诊断，不用它们选正式策略。训练可去掉 `--replay` 续跑，每 100
步写检查点；查看运行目录的 `heartbeat.json` 和 PID 确认进程状态。

```sh
.venv-pytorch/bin/python scripts/run_m3w_cv_lead_mixture.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_lead_mixture_capacity.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_minimum_edit_count.py
```

应得到 2,317 个固定改动、两组校准均为 0/9、候选补入后仍只有 3/72 个
事后双基线可行视图。忽略风险、授予完美预测的上界在原预算下排除 64/72。
这不是调整原 2% 风险规则，也不允许解封确认集。
[v129 编译说明](../outputs/publication_readiness_2026_09/evidence_manuscript_v129/README.md)。

## v128：录像级分配和因果额度结转

用本机冻结候选和拟合来源重放。第一个入口会复核选择冻结和录像评分，第二个
重算邻居相对位移误差，第三个复核标签可见的事后可行方案。离线重排需要整个
录像的预测分数，不能冒充在线；第三个入口使用结果标签，不能用来选测试策略。

```sh
.venv-pytorch/bin/python scripts/run_m3w_cv_record_allocation.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_record_allocation_geometry.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_record_allocation_capacity.py
```

同一录像预算合计 2,317 次；原分配、离线收益、离线风险惩罚分别通过
14/72、4/72、9/72。在线校准 0/9。事后允许挪额度后，满足原规则且同时
超过 CV 与 OLS4 的视图为 3/72。[v128 编译说明](../outputs/publication_readiness_2026_09/evidence_manuscript_v128/README.md)。

## v127：先筛候选与成对排序

有本机冻结的拟合来源数据、风险模型和来源隔离预测时，下面两个入口只用开发
角色。第一个复算筛选顺序，第二个加载训练检查点并精确复算选择和评分，不更新
参数。移除 `--replay` 会从现有检查点续跑，每 100 次更新保存一次；先检查
`data/stage_cvpr2027_experiments/moving_cv_paired_nomination_v1/heartbeat.json`
中的 PID 和训练步，再决定是否恢复。

```sh
.venv-pytorch/bin/python scripts/run_m3w_cv_screen_order.py
.venv-pytorch/bin/python scripts/run_m3w_cv_paired_nomination.py --replay
```

匹配预算为每组 2,317 次改动；两种排序都只通过 14/72 个录像视图，校准均为
0/9。已知 easy 最差退化 0.84% 不等于完整 2% 风险门槛通过。论文构建命令见
[v127 说明](../outputs/publication_readiness_2026_09/evidence_manuscript_v127/README.md)。

## 本机环境

在项目根目录运行。已核验环境为 Apple Silicon `arm64`、Python 3.11.1、
Torch 2.12.0、NumPy 2.4.6、SciPy 1.17.1、Matplotlib 3.10.9。不要用 Rosetta
下的 x86_64 Conda 环境代替。训练入口使用四个计算线程和单进程采样；没有
DataLoader 子进程。版本检查只确认环境，不代表训练已经跑通。

```sh
.venv-pytorch/bin/python -c 'import platform, torch, numpy; print(platform.machine(), torch.__version__, numpy.__version__)'
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_motion_choice_cost.py tests/test_m3w_motion_cost_stability.py tests/test_m3w_cv_anchor_budget.py
```

## 重现论文中的改动预算结论

有权限使用本机已审核的 A 数据与原检查点时，可运行下面的零更新复跑。GitHub
代码和轻量结果不包含这些原始数据与私有模型；缺少它们时不能声称从原始数据
重现。第一个入口复现模型选择和评分，后两个入口重新计算解析上界。

```sh
.venv-pytorch/bin/python scripts/run_m3w_motion_cost_stability.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_sparse_anchor_risk.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_anchor_budget.py
```

应核对：27 个来源留一检查点的零更新复跑；OLS4 起点的单轨改动在 34/34 个
录像上不可能过原风险规则；CV 起点的单轨任意改动在 30/34 个录像上不能获得
相对 OLS4 的正保守收益；有限候选加宽松风险上界后为 34/34。
可见误差、缺失步骤的保守界限和抽样区间不可互相替代。

扩展候选的联合可行性实验可用下列入口复核。`--replay` 重建并验证保存的轨迹
选择，不重做求解；没有本机缓存时不能声称已经复核。重新运行不带 `--replay`
的入口只会补齐尚未完成的求解格。确认新增视图是 15（上限 8）和 12（上限 32）。

```sh
.venv-pytorch/bin/python scripts/audit_m3w_cv_joint_feasibility.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_joint_matched_shape.py
.venv-pytorch/bin/python scripts/build_m3w_cv_joint_replay.py
```

匿名包在私有 `moving_cv_joint_feasibility_v1` 目录下。解压到空目录后用
对应 Python 运行 `replay.py`；它校验哈希和匿名汇总，不读取原始轨迹。

新的选择器实验先在 9 个拟合来源内部交叉训练预测器，再训练单轨和成对选择头。
首次从头运行需预留约 25 分钟本机计算；完整 `--replay` 不更新参数。已有冻结
权重时核对 18 个来源排除模型以及 2,304 条重算预测，再重放选择与评分：

```sh
.venv-pytorch/bin/python scripts/verify_m3w_cv_joint_oof_bank.py
.venv-pytorch/bin/python scripts/run_m3w_cv_joint_selector.py --replay
.venv-pytorch/bin/python scripts/build_m3w_cv_joint_selector_replay.py
```

每个种子每个策略改动 12,896 条。相加对照与联合选择共享候选、权重和改动数，
原风险通过均为 2/102。重放并不证明它通过安全要求。

v126 的可选改动只在原拟合来源内部轮换训练、校准和留出三种用途。复跑不更新
参数。后两个入口使用留出结果计算事后上限，不是部署方法：

```sh
.venv-pytorch/bin/python scripts/run_m3w_cv_admission.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_admission_capacity.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_admission_unpruned.py
```

9 个校准组合均找不到合格设置。保留 CV 不等于 2% 风险检查通过，因为未发生
改动时，所需的参考量不可估。保留 OLS4 的 15/72 是事后可行性，不是模型实绩。

需要完成中断的本机训练时，先看该实验私有目录中的 `heartbeat.json`，核对
PID 对应的进程确实退出，再用原命令去掉 `--replay`。训练每 100 步原子保存
模型、优化器与采样状态，身份或数据哈希不同会报错。不要删除有效检查点。

## CREATE：已有工作怎么核对

这组实验在本机即可完成，不需要提交 CREATE。先看
[最近的调度与哈希核验](../outputs/publication_readiness_2026_09/create_accounting_recovery_20261108/README.md)。
针对其他已获准的 M3W 任务，只使用 M3W 的目录和已核对环境。先查询已知作业的
`squeue`（活动状态）和 `sacct`（终态、退出码与资源），核对既有作业和配额，
再通过独立的 M3W 作业脚本在计算节点运行。登录节点只做只读核对。一个已核对
的历史入口及其提交、续跑和结果收集方式见
[时序实验 CREATE 复现说明](../outputs/publication_readiness_2026_09/european_temporal_auxiliary_v1/create_reproduction_zh.md)；
里面的历史提交命令不是本轮的重跑命令。网络超时不表示作业终止。

训练输出至少需要作业 ID、运行日志、环境版本、训练步数和检查点哈希。恢复前
核对相同训练身份与采样状态；完成后再核对实际预测、数值、数据角色和各来源
结果。调度成功或合成环境测试不代表科研指标通过。

## 构建论文

参见 [v130 构建与校验](../outputs/publication_readiness_2026_09/evidence_manuscript_v130/README.md)。
主文、补充材料与生成图的命令均由对应的冻结读出生成。匿名复现只展示可公开的
汇总量，不包含原始轨迹、视频、模型权重或本机路径。当前工作稿没有提交。
