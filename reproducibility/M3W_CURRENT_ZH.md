# M3W 当前论文与关键实验：中文复现入口

本文档对应 v148 研究稿。实验结果分三类：开发录像上的实测、使用未来标签的
事后上界、独立确认。前两类都不能写成第三类。DroneCrowd 的 15 个校准来源组
没有得到满足既定规则的策略，30 个独立确认来源组不用于以下命令。

## v148：直接训练因果速度预测器

```sh
.venv-pytorch/bin/python scripts/train_m3w_causal_velocity_spline.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_causal_velocity_spline.py
.venv-pytorch/bin/python scripts/audit_m3w_causal_spline_capacity.py
.venv-pytorch/bin/python scripts/audit_m3w_spline_ols_tail.py
```

去掉第一条的 `--replay` 可从已有优化器检查点恢复或重新训练缺少的模型。
需要原有开发角色、冻结神经对照，以及 v147 的本地容量凭证。第一条只在
验证角色选版本，先冻结预测再评价；第二条分解已观测误差、未观测扣减和
双人相对位移；第三条事后使用结果检验同改动数候选容量，不是在线选择器。
第四条在 OLS4 周围匹配每条轨迹的总位移，在验证角色选择收缩方式；测试切片
不参与选择。它属于看过开发结果后的探索，不是独立验证。
[v148 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v148/README.md)。

## v147：OLS 锚定路径的等改动数容量检验

```sh
.venv-pytorch/bin/python scripts/audit_m3w_ols_anchor_menu.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_ols_anchor_budget_location.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_cv*.py tests/test_m3w_ols_anchor_menu.py
```

去掉 `--replay` 可重新求解或续跑。需要 v146 的本地冻结决策和开发录像输入。
原候选集与扩展候选集使用完全相同的每查询改动数和原录像级 2% 判据。候选轨迹
只从过去生成，但求解看到了结果，因此这一步不是在线预测或模型训练。
超时不计为不可行；中断的预处理试跑与正式结果分开保存。
第二条只算去除风险约束后的 OLS4 上界：36 个已排除的视图中，即便保持录像总
改动数、允许移动改动位置，28 个仍无法取得正上界。正式风险规则未改。
[v147 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v147/README.md)。

## v146：固定预测风险与改动数，学习准确率收益

```sh
.venv-pytorch/bin/python scripts/train_m3w_cv_fixed_budget_gain.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_fixed_budget_capacity.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_cv_fixed_budget_gain.py
```

去掉 `--replay` 可训练或从每 100 步的检查点续跑。需要 v145 已核验的
本地检查点和冻结路径选择；Git 不保存权重和逐行决策。冻结每个查询的改动数、
预测风险、参考支持和 CV 收益预算，比较直接收益回归、分解收益回归和几何对照。
事后搜索与分数松弛上界只用于诊断，不参与策略选择。上界为平均 +0.0577 pp，
不足以追平 OLS4。原校准通过 0/9，部署不变。
[v146 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v146/README.md)。

## v145：拆分正向伤害、缺标和 easy 参考

```sh
.venv-pytorch/bin/python scripts/train_m3w_cv_risk_components.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_risk_component_swaps.py
.venv-pytorch/bin/python scripts/eval_m3w_cv_risk_component_low_count.py
```

首条从冻结检查点复算，去掉 `--replay` 可训练或续跑。第二条逐项替换风险估计；
第三条把每个查询的改动数固定在联合策略的实际低覆盖度。直接独立、分项独立、
分项联合分别通过 2/72、1/72、20/72，但联合没有保住保守准确率，校准 0/9。
[v145 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v145/README.md)。

## v144：显式累积录像前缀状态

```sh
.venv-pytorch/bin/python scripts/train_m3w_cv_prefix_state.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_prefix_state.py
.venv-pytorch/bin/python scripts/eval_m3w_cv_prefix_arithmetic.py
```

第一条从冻结权重重算，去掉 `--replay` 可训练或续跑。第二条诊断收益与伤害的预测
误差。第三条不训练新模型，直接比较有无历史余额。相同改动数下，两个状态对照
均无保守增益；学习组都只通过 3/72。确认组保持封存。
[v144 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v144/README.md)。

## v143：用同改动数的边界样本训练

```sh
.venv-pytorch/bin/python scripts/train_m3w_cv_risk_crossing.py --replay
.venv-pytorch/bin/python scripts/build_m3w_cv_risk_crossing_replay.py
```

第一条从冻结检查点重算；去掉 `--replay` 可以训练或续跑。第二条复算原始计分并
生成匿名标量复现包。风险监督在匹配改动数下把最差 easy 有符号增幅从
48.80% 降至 23.59%，但原规则均只通过 1/72，保守准确率也下降。
[v143 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v143/README.md)。

## v142：用预测风险预算生成整组候选

```sh
.venv-pytorch/bin/python scripts/audit_m3w_cv_risk_frontier.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_risk_frontier_integer.py
.venv-pytorch/bin/python scripts/eval_m3w_cv_risk_frontier_transfer.py
```

前两条是事后容量诊断：8/72、11/72、27/72 分别是原菜单、扩展菜单、同路径库
逐智能体自由组合的已验证数；后者另有 24 个未求解。第三条用冻结模型和独立的
校准角色比较扩展前后，未得到可部署方案。两条长任务保留本机断点与心跳。
[v142 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v142/README.md)。

## v141：让模型选择整组改动和数量

```sh
.venv-pytorch/bin/python scripts/train_m3w_cv_joint_count.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_joint_count_capacity.py
```

第一条从冻结检查点重算；去掉 `--replay` 可以训练或续跑。第二条在相同候选
上做事后容量诊断。学到的联合策略在等改动数量下不如独立选择；放开数量把事后
符合原规则的视图从 0/72 增至 8/72，但不能据此部署。
[v141 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v141/README.md)。

## v140：对实际候选训练伤害分位数

```sh
.venv-pytorch/bin/python scripts/train_m3w_cv_selected_harm.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_selected_harm_capacity.py
```

先用冻结检查点重算，再算固定候选和固定改动数的事后上界。去掉
`--replay` 可训练或续跑；先核对 PID 和心跳。两种训练各用同样样本，
只改损失的分位数。在相同改动数下，高分位模型没有胜过中位数模型；
全候选仍在 45/72 个视图无法胜过 OLS4，放开改动数后该项排除降为 0/72。
后者不含风险约束，不是部署结果。[v140 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v140/README.md)。

## v139：学习改动幅度

```sh
.venv-pytorch/bin/python scripts/train_m3w_cv_amplitude_response.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_amplitude_response.py
.venv-pytorch/bin/python scripts/build_m3w_cv_amplitude_response_replay.py
```

第一条从冻结检查点复算，去掉 `--replay` 可训练或续跑；先核对 PID 和心跳。
第二条比较曲线误差，区分已观测伤害和缺标带来的不确定性。第三条生成匿名标量
复算 ZIP。等总峰值位移下，可见部分有 +0.071 pp，保守下界却是 -0.380 pp；
校准 0/9，未开启确认组。[v139 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v139/README.md)。

## v138：同改动数和同峰值预算的对象交换

```sh
.venv-pytorch/bin/python scripts/train_m3w_cv_matched_exchange.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_exchange_peak_envelope.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_exchange_record_pool.py
.venv-pytorch/bin/python scripts/build_m3w_cv_matched_exchange_replay.py
```

训练脚本使用本机 arm64 环境和检查点；移除 `--replay` 可训练或续跑，先核对 PID。
后两种预算诊断允许事后最有利的路径，不用于选阈值。同一录像总峰值预算下，
任意换对象也应重现 67/72 个 OLS4 排除。最后一个命令生成匿名标量 ZIP，
只用 NumPy 即可复算风险和收益表。[v138 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v138/README.md)。

## v137：固定对象、等位移和来源一致性

```sh
.venv-pytorch/bin/python scripts/train_m3w_cv_fixed_recipient_support.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_fixed_recipient_envelope.py
.venv-pytorch/bin/python scripts/build_m3w_cv_fixed_recipient_support_replay.py
```

第一条用冻结权重复算，去掉 `--replay` 可训练或续跑；先检查 PID 和心跳。
第二条算允许任意路径的事后上界，不用它选择阈值。第三条生成匿名 NumPy
标量复算 ZIP。54,748 个编辑对象固定，半幅策略均有 18/72 个风险通过，
校准 0/9。任意路径在无位移限制时仍有 39/72 个视图无法胜过 OLS4。
[v137 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v137/README.md)。

## v136：改动对象与路径分别学习

```sh
.venv-pytorch/bin/python scripts/build_m3w_cv_recipient_teachers.py
.venv-pytorch/bin/python scripts/train_m3w_cv_recipient_path.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_recipient_radius.py
.venv-pytorch/bin/python scripts/build_m3w_cv_recipient_path_replay.py
```

第一条生成或复核训练来源里的 237 套可行方案。第二条固定权重重算；去掉
`--replay` 可从检查点训练，先核对 PID 和心跳。第三条固定对象与峰值位移，
第四条生成只需 NumPy 的匿名标量复算包。等位移保守差值应为 -1.133 pp，
原规则校准 0/9。[v136 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v136/README.md)。

## v135：最小改动预算与教师模型

```sh
.venv-pytorch/bin/python scripts/run_m3w_cv_minimum_budget.py teacher --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_budget_certificates.py
.venv-pytorch/bin/python scripts/train_m3w_cv_minimum_budget.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_minimum_budget_failures.py
.venv-pytorch/bin/python scripts/build_m3w_cv_minimum_budget_replay.py
```

教师与训练命令去掉 `--replay` 后可续跑；先检查本地心跳和 PID，勿启动重复进程。
保持原规则时，25% 录像预算有 38/72 个事后见证。相同查询改动数的教师对照仍失败，
校准 0/9。匿名 ZIP 在干净目录解压后以 NumPy 运行 `replay.py` 可复核标量表，
不需要轨迹或权重。[v135 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v135/README.md)。

## v134：候选范围与查询改动数量的四组对照

```sh
.venv-pytorch/bin/python scripts/audit_m3w_cv_budget_menu_factorial.py --replay
.venv-pytorch/bin/python scripts/verify_m3w_cv_budget_menu_inclusion.py
.venv-pytorch/bin/python scripts/build_m3w_cv_budget_menu_replay.py
```

第一条核验本地求解见证；去掉 `--replay` 可以续跑缺失部分。第二条核验放宽
限制后仍成立的见证，第三条生成匿名 ZIP。解压后只用 NumPy 运行
`replay.py` 可复核四组计数和分组区间，不需要轨迹或权重。按窄/完整候选
与固定/自由数量交叉，已验证计数是 5/19/5/53，另有 13 个条件内视图未决。
这不是策略泛化结果。[v134 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v134/README.md)。

## v133：整组改动监督与固定配额容量

```sh
.venv-pytorch/bin/python scripts/build_m3w_cv_slate_teachers.py
.venv-pytorch/bin/python scripts/run_m3w_cv_slate_supervision.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_fixed_slate_capacity.py
```

第一条只在各折训练来源内构造或核验整录像教师。第二条核验冻结权重、逐查询
改动数量和结果，不更新参数；首次训练或确认旧进程已退出后续跑，去掉
`--replay`。第三条只做事后诊断，固定每个查询的原改动数量，不参与模型或
阈值选择。应复现 54,748 次等数量改动、校准 0/9、固定菜单可行 5/72。
[v133 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v133/README.md)。

## v132：逐步缺失概率、easy 风险和等位移对照

```sh
.venv-pytorch/bin/python scripts/run_m3w_cv_lead_risk.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_lead_risk_evidence.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_lead_disagreement.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_late_edit_taper.py
```

主实验四组逐查询匹配，均为 51,614 次改动。第一条仅复算冻结模型；如需新训或
确认进程已退出后续跑，去掉 `--replay`。第二条生成匿名标量 ZIP，解压后运行
`replay.py` 无需轨迹或 Torch。后两条复核缺失位移归因和等总位移对照；
等总位移不等于等峰值，也不构成合格策略。逐步标注概率改善 Brier，不代表
保守收益改善。校准仍为 0/9。
[v132 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v132/README.md)。

## v131：有符号风险余量与等数量收益排序

```sh
.venv-pytorch/bin/python scripts/run_m3w_cv_signed_coverage.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_signed_matched_gain.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_signed_failures.py
```

第一条命令核对冻结检查点、重算预测和校准结果，不更新参数。首次训练或确认
原进程已退出后续跑，去掉 `--replay`；每 100 步写入模型、优化器和采样状态。
第二条用同一风险模型，在每个查询严格匹配介入数；第三条把已知 easy 伤害和
缺失标签导致的额外失败分开，不能把“只看已知”当成通过原规则。

条件风险模型在同一真实候选上识别 1,843/2,453 个机会，校准仍为 0/9。
条件风险下，两种收益排序各改 71,595 条；应重现已知收益 +0.2686 pp 和
保守下界差 −1.3641 pp。[v131 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v131/README.md)。

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

参见 [v132 构建与校验](../outputs/publication_readiness_2026_09/evidence_manuscript_v132/README.md)。
主文、补充材料与生成图的命令均由对应的冻结读出生成。匿名复现只展示可公开的
汇总量，不包含原始轨迹、视频、模型权重或本机路径。当前工作稿没有提交。
