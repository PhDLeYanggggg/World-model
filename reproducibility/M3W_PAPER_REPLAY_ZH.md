# M3W 论文：本机与 CREATE 复现入口

当前候选：[v200 主文构建说明](../outputs/publication_readiness_2026_09/evidence_manuscript_v200_wepdtof_history/README.md)。
这份短教程区分匿名汇总复算、真实检查点复跑和重新训练。三者的证据范围不同。

另有未评分的 [Athens / Swiss 九格对照](../outputs/publication_readiness_2026_09/external_recipient_factorial_v1/README.md)：
两地只跑过去轨迹推理与人工目标排演，可在批准正式划分前复现；不得把它写成真实外部精度。

## 1. 匿名补充包：只复算已冻结结果

本机补充包为 `data/stage_cvpr2027_experiments/submission_admin/review_supplement_v31.zip`。
解压到空目录，从该目录运行 `python3 -I -B -S verify.py`（Python 3.10 或更新版本）。
不需要网络、权重或原始轨迹。应看到 Moving 三种上界排除数 67、36、9 / 72；
OpenDD 330/606 对 57/606；WEPDTOF 0/45 同时通过风险和效用检查；Songdo 联合策略 0/480 放行。
v4 另复算固定次数和位移后 OpenDD 的 210、60、42 三个关键计数和地点区间。
v5 保留这些结果，并复算成对偏好、等步数续训和风险内收益续训。
v6 新增编辑交换界限：零交换 60、至多一次 178–183、不限次数 210–212，另复算三个描述性收益门槛。
v7 保留这些界限，并将至多一次的范围精确收紧为 181；另复算同录像随机对照。
v8 增加地图和邻居消融：在同样 101 次交换上复算正确对齐 47 个双通过、
随机期望 51.21，以及对应的地点置信区间。主文仍为 v179。
v9 配套 v179，保留这些检查并加入六项方向对照：校准把旋转从 710 降到 198，
固定角度在同一批对象上优于学习角度；同数量随机对象复现了大部分收益。
v10 配套 v180，保留 v9 并加入独立的车辆开发对照：493 段录像的 8,874 个重复视图中，
1,431 个事后有合格方向解且相对运动改善，1,393 个只转一辆就有解。
这些车辆计数不与行人计数合并。
v11 配套 v181，另复算真实车距修正和等模型邻车消融。以相同的 5,162 个旋转对象作比较，
带邻车的角度选择仍未赢得保守轨迹收益。
v12 配套 v182，另复算直接学习收益与伤害的对照。在同样 2,399 个旋转视图上，
模型优于供体固定角度，但不如单车模型。
v13 配套 v183，保留上述结果并复算固定候选动作的录像筛选。在相同 526 次旋转预算下，
原置信度与新筛选器分别兑现 54、41 个严格通过。
v14 配套 v184，新增精确动作分类与预算排序，严格通过分别为 41、40。
v15 配套 v185，新增路径族对齐，同样 526 次出手下对齐后的严格通过为 35、44。
v16 另复算策略选车和前 3 候选的事后上限；同预算下的上限从 238 升至 329。
v17 固定原出手录像复算 54、50、55、56 个严格通过，事后上限 85；
固定 15% 预算下轨迹误差略改善，相对运动变差。校准后没有替换原策略。
v18 另复算在原 526 次出手中，66 次有事后合格替代，预测一致规则选中九次，
跨地点校准后换角一次，没有合格改善。出手录像未变。
v19 另复算直接监督对同网络代理标签的对照：44,370 个训练对中，
8,412 个相对运动的正收益标签不同；校准后无合格替换。包内同时核对缺失标签界限。
v20 复算 Songdo 缺失标签的上下界及来源 bootstrap，并与 WEPDTOF 的冻结结果对照。
两种区间不能混称置信区间，也不改变原风险结论。
v21 保持 v190 主文不变，新增 OpenDD 共用缺失位置的分析。固定 526 次出手时，
直接监督与锚点差分的比较有 83 个视图跨零，其中 82 个展示了正、负两种补全。
补充包核对汇总计数并复算区间，不包含用于构造补全的逐帧位置。
v22 配套 v191，按七个地点复算等换角次数的对照：每边 78 次替换，
两种模型分别得到 11 和 13 个合格替换，区间模型自身的随机子集期望是 13.98。
这是旧开发集的等预算复算，不是新训练或独立确认。
v23 配套 v192，按 14 个来源组复算 Songdo 三类介入对象的误差贡献，并核对它们
相加后严格复现原主结果。归档不含未来坐标、模型权重或原始录像。
v24 配套 v193，固定被介入车辆，复算两种选路规则的差异，并核对两条交叉路径的
已观测误差加总都回到原对照。这不是新独立测试。
v25 配套 v194，复算 OpenDD 固定 353 次换车的对照、145/2,624 的机会覆盖，
以及按地点配对的轨迹收益区间。这里复算的是汇总，不是重新训练。
v26 配套 v195，保留 v25，复算重新分配 353 次换车的比较、126 组预算和地点配对区间。
v27 配套 v196，复算同一批候选换车的 68 / 47 / 43 次合格数、已观测相对收益与缺失标签界。
v28 配套 v197，复算可观测支持的排序提升、原预算下 68 / 55 / 54 次合格数及地点配对区间。
v29 配套 v198，复算 126 个校准/留出/种子组合、放行数、旧对照和地点区间；空放行不是风险通过。
v30 配套 v199，复算固定候选后的 32 / 58 / 2 次放行、6 / 9 / 0 次严格通过，
以及 3,000 次地点配对重抽样。只复算汇总，不是重新训练或新外部确认。
v31 配套 v200，复算 WEPDTOF 的 353 条匹配历史中，起点和速度对 ADE 的贡献，
以及 OLS4 截距重估的 13 来源配对区间。原 v30 一并保留。
四种冻结方案中 197/480 事后有合格方案的补充分析，从项目根目录运行：

```sh
python3 -I -B -S scripts/audit_m3w_songdo_frozen_menu_capacity.py
```

这不是重新推理或读标签；197/480 不是可部署成绩。

## 2. 本机：来源核对与真实检查点复跑

使用 Apple Silicon 原生 `.venv-pytorch/bin/python`。先核对解释器架构、Torch 和 NumPy；
不要用 Rosetta 下的 x86_64 Conda。训练入口设 4 个 Torch 计算线程、单进程采样，
不创建 DataLoader 子进程。若要换硬件或重建环境，先用冻结检查点跑通，再解释训练日志。

```sh
.venv-pytorch/bin/python -c 'import platform, torch, numpy; print(platform.machine(), torch.__version__, numpy.__version__)'
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_paper_evidence_lineage.py
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/run_m3w_recording_count.py --replay
```

来源核对应找到 7 个 OpenDD 地点、202 段录像和 3,395 个录像内不重叠查询；
OpenDD 检查点复跑应完成全部 21 个地点×种子组合，终态 `complete`，且 `updates: 0`。
它会读本机已经使用过的开发目标，但不会打开封存的 DroneCrowd 确认标签。
Git 里的轻量文件不含私有权重和原始数据。缺少冻结权重就不能声称复跑成功。

v200 的 WEPDTOF 回顾性实验运行 `scripts/audit_m3w_wepdtof_history_factorial.py`。
这条命令读回已评分的 15 段 WEPDTOF 标注，核对三种子原 CV/OLS4 指标，
再评分固定的历史替换对照。不是匿名汇总复算，也不会打开雅典、瑞士或 DroneCrowd 封存目标。

v199 的 easy 参考实验依次运行
`scripts/run_m3w_opendd_easy_reference_release.py freeze`、`calibrate`、`score`。
沿用已发布的分数和阈值时，会核对冻结文件及模型校验值。缺少训练输出时，
只要原始训练输入和原冻结候选还在，训练可以从每 100 步的断点恢复。
[结果与命令](../outputs/publication_readiness_2026_09/opendd_easy_reference_release_v1/README.md)。

复算 v180 的车辆方向实验，依次运行
`scripts/audit_m3w_vehicle_angular_pair.py` 和 `scripts/audit_m3w_vehicle_single_rotation.py`。
两项沿用本机冻结的车辆双编辑记录，不重新训练或读外部确认标签。
[双车对照](../outputs/publication_readiness_2026_09/opendd_vehicle_angular_pair_v1/README.md)、
[同对象单旋转对照](../outputs/publication_readiness_2026_09/opendd_vehicle_single_rotation_v1/README.md)。

v181 的真实车距修正使用 `scripts/run_m3w_vehicle_direction_context.py freeze` 冻结所有选择，
再用同脚本的 `score` 评分。需要本机已核验的车辆输入及坐标对齐文件。
训练断点每 50 步保存；有完整冻结文件时只复核检查点和选择文件，不重训。
[实验、命令及区间](../outputs/publication_readiness_2026_09/opendd_vehicle_direction_context_v1/README.md)。

v182 的直接动作收益模型使用 `scripts/run_m3w_vehicle_action_value.py freeze` 与 `score`。
另用 `scripts/audit_m3w_vehicle_action_capture.py` 重算事后归因：
1,393 个严格可改善视图中，未旋转、选错车辆、选错角度、兑现改善分别为 855、96、281、161。
[训练、校准和归因](../outputs/publication_readiness_2026_09/opendd_vehicle_action_value_v1/README.md)。

v183 的录像筛选使用 `scripts/run_m3w_vehicle_opportunity.py freeze` 和 `score`。
它只改变哪些录像使用预先确定的动作，不改变旋转对象或角度。
[训练、冻结、对照和置信区间](../outputs/publication_readiness_2026_09/opendd_vehicle_opportunity_v1/README.md)。

v184 先运行 `scripts/fit_m3w_vehicle_nested_nominees.py` 核验或生成内层留源权重，
再用 `scripts/run_m3w_vehicle_nominee_rank.py freeze` 与 `score`。
用 `scripts/audit_m3w_vehicle_nominee_anchor_shift.py` 重放固定候选的路径族统计；
该脚本不读取未来坐标数组。
[完整训练、校准和结果](../outputs/publication_readiness_2026_09/opendd_vehicle_nominee_rank_v1/README.md)。

v185 使用 `scripts/run_m3w_vehicle_family_alignment.py freeze` 与 `score`。
复用 v184 的嵌套权重、原始训练抽样和留出提名，不需要重新生成外部候选。
[训练和对齐结果](../outputs/publication_readiness_2026_09/opendd_vehicle_family_alignment_v1/README.md)。

v186 依次运行 `scripts/fit_m3w_vehicle_nested_additive.py`、
`scripts/run_m3w_vehicle_policy_anchor.py freeze` 与 `score`。
`scripts/audit_m3w_vehicle_nominee_topk.py` 只复算已冻结候选的事后上限，不训练新策略。
[训练、配对结果和候选上限](../outputs/publication_readiness_2026_09/opendd_vehicle_policy_anchor_v1/README.md)。

v187 使用 `scripts/run_m3w_vehicle_shortlist.py freeze` 与 `score`，
再用 `scripts/audit_m3w_vehicle_shortlist_fixed_mask.py` 复算固定出手掩码的事后上限。
[训练和同预算对照](../outputs/publication_readiness_2026_09/opendd_vehicle_shortlist_v1/README.md)。

v188 使用 `scripts/run_m3w_vehicle_angle_agreement.py freeze` 与 `score`；
这两条复用 v187 本机权重，不训练新模型。
[规则、复跑和区间](../outputs/publication_readiness_2026_09/opendd_vehicle_angle_agreement_v1/README.md)。

v189 使用 `scripts/run_m3w_vehicle_pair_angle.py freeze` 和 `score`。
前者按已完成的方向和训练检查点恢复，后者复用已评分的方向；另用
`scripts/audit_m3w_vehicle_pair_angle_bounds.py` 复算相对运动的两侧界限。
[配对训练与复算](../outputs/publication_readiness_2026_09/opendd_vehicle_pair_angle_v1/README.md)。

要复算 v175 新增的 OpenDD 固定实际编辑次数诊断，用
[`audit_m3w_opendd_realized_stream_capacity.py` 和 `report_m3w_opendd_realized_stream_capacity.py`](../outputs/publication_readiness_2026_09/opendd_realized_stream_capacity_v1/README.md)。
需要本机冻结权重和原先预算上限计算的逐录像记录；终态应有 606 个视图、780 次实际编辑，
其中固定次数事后解 242 个通过，线上 57 个通过。这不是匿名 v3 包中的检查。

随后运行 `scripts/audit_m3w_opendd_count_motion.py`，在每段录像固定编辑次数，
同时限制累计峰值位移。事后重排分数 60/606、已知标签事后解 210/606；
只重排分数不使用标签，但能看全段分数，因此不是线上部署。
[数值与复跑](../outputs/publication_readiness_2026_09/opendd_count_motion_v1/README.md)。

新训练的评分头用 `scripts/run_m3w_opendd_prefix_value.py --replay` 核对 42 个检查点，
断点恢复时去掉 `--replay`。固定预算结果为旧头 60、只学增益 48、同一联合头不带惩罚 57、
带惩罚 42。它没有通过开发对照，不能替换旧策略。
[训练与负结果](../outputs/publication_readiness_2026_09/opendd_prefix_value_v1/README.md)。

等预算方案排序用 `scripts/run_m3w_opendd_equal_spend_preference.py --replay` 复核。
同样的 780 次编辑，旧分数 60、独立分类 54、成对排序 59 个双通过；
排序增加了风险通过数，但没有增加同时满足收益和风险的方案。
[完整对照与训练记录](../outputs/publication_readiness_2026_09/opendd_equal_spend_preference_v1/README.md)。

风险内收益续训用 `scripts/run_m3w_opendd_risk_conditional_utility.py --replay`。
普通续训 55/606、风险内收益续训 58/606、旧分数 60/606；每组仍是相同 780 次编辑。
[训练与区间](../outputs/publication_readiness_2026_09/opendd_risk_conditional_utility_v1/README.md)。

交换范围用 `scripts/audit_m3w_opendd_exchange_frontier.py` 和
`scripts/report_m3w_opendd_exchange_failures.py`。第一条在旧方案周围找事后最优，
第二条分解原风险与收益失败。结果的上下端是已验证与未排除的计数，不是置信区间。
[完整计数与说明](../outputs/publication_readiness_2026_09/opendd_exchange_frontier_v1/README.md)。

排序器用 `scripts/run_m3w_opendd_one_exchange_ranker.py --replay` 复核；
`scripts/audit_m3w_opendd_one_exchange_ranking.py` 穷举 7,198 个交换，复算同录像随机对照。
不重新训练，不使用未打开的外部数据。[训练与读数](../outputs/publication_readiness_2026_09/opendd_one_exchange_ranker_v1/README.md)。

地图和邻居对照用 `scripts/run_m3w_opendd_exchange_context.py --replay`；
`scripts/audit_m3w_opendd_exchange_context_matched.py` 固定相同的交换视图并计算精确随机期望。
[输入、训练和复算](../outputs/publication_readiness_2026_09/opendd_exchange_context_v1/README.md)。

在本地开发数据上，`scripts/audit_m3w_opendd_exchange_attrition.py` 区分漏出手与选错交换；
可挽救 121 个，其中 77 个没有出手。随后用
`scripts/run_m3w_opendd_exchange_risk_heads.py --replay` 核对显式风险监督对照：
同样的 101 次交换下为 49 个双通过，未超过随机期望 51.21。
[检查点与训练命令](../outputs/publication_readiness_2026_09/opendd_exchange_risk_heads_v1/README.md)。

运行 `scripts/audit_m3w_opendd_exchange_stability.py` 可复算保护交换两端的删段诊断：
121 个可挽救视图中 93 个存在稳定交换。再运行
`scripts/run_m3w_opendd_exchange_stability_head.py --replay` 和
`scripts/audit_m3w_opendd_exchange_stability_forced.py`，分别复算固定 101 个交换视图和强制 296 个合法视图。
两项都不重新训练；新监督没有可靠胜过同架构对照。

三项形状比较依次运行 `scripts/audit_m3w_opendd_exchange_phase.py`、
`scripts/audit_m3w_opendd_exchange_matched_amplitude.py` 和
`scripts/audit_m3w_opendd_exchange_amplitude_control.py`。
终点、峰值位移及出手记录固定。累计偏离对齐后，没有额外的推迟偏离收益。
[完整对照](../outputs/publication_readiness_2026_09/opendd_exchange_amplitude_control_v1/README.md)。

改变偏离方向、逐步位移不变的对照，用 `scripts/audit_m3w_opendd_exchange_directions.py`；
事后双通过上限应为原 181、扩展 216。神经方向头用
`scripts/run_m3w_opendd_exchange_angular_head.py --replay`，零更新复算 101 和 296 两种出手范围。
再运行 `scripts/audit_m3w_opendd_exchange_angular_debit.py` 与
`scripts/audit_m3w_opendd_exchange_angular_same_recipients.py`，分开核对缺失步扣减和原 780 个编辑对象。
[训练和结果](../outputs/publication_readiness_2026_09/opendd_exchange_angular_head_v1/README.md)。

固定原编辑对象，对照原方案损失可用
`scripts/run_m3w_opendd_exchange_angular_incumbent.py --replay`；
另用 `scripts/audit_m3w_opendd_angular_incumbent_oracle.py` 复算 13 个角度的事后效用上限。
实际可保留原方向的模型用 `scripts/run_m3w_opendd_angular_choice.py --replay`：
拆分模型应为 70 个双通过，但没有证实保守效用提升。
[训练和检查点](../outputs/publication_readiness_2026_09/opendd_angular_choice_v1/README.md)。

内层校准用 `scripts/run_m3w_opendd_angular_calibration.py calibrate --replay`；
外层只评分用同脚本的 `score`。门槛提交 `1d285ab78` 先于外层评分。
再运行 `scripts/audit_m3w_opendd_angular_calibration_mask.py` 检查同录像、同旋转数的精确随机对照。
[校准协议和复算](../outputs/publication_readiness_2026_09/opendd_angular_calibration_v1/README.md)。

OpenDD 入口每 100 步原子保存模型、优化器、采样状态和训练身份。仅在确认既有
PID 退出、数据和注册文件一致后，才用原命令去掉 `--replay` 恢复训练；不要删除有效
检查点。恢复后先看 `heartbeat.json` 与实际日志，再核对结果及哈希。

区间监督消融有单独的[预注册](../outputs/publication_readiness_2026_09/opendd_vehicle_interval_angle_v1/protocol.md)。
使用相同的本机线程设置运行 `scripts/run_m3w_vehicle_interval_angle.py freeze`，
中断后运行同一条命令继续。冻结文件与仓库中的版本一致后，再用 `score` 计算开发集对照；
只用对侧地点定阈值，不能当作一次新的独立外部检验。

雅典 pNEUMA 的[只读准备与复现](../outputs/publication_readiness_2026_09/pneuma_prefix_pilot_v1/README.md)
生成 643 个历史窗口的三种子预测，并固定每个对照两个等位移的候选动作。
分块搜索与原搜索一致。此处不打开未来坐标、不计算准确度，也不决定正式划分。
用 `scripts/rehearse_m3w_pneuma_confirmation.py` 可以在合成数据上复跑两种角色协议，
用 `scripts/prepare_m3w_pneuma_donor_cuts.py` 可以核对纯 OpenDD 的候选难度切线。
[实际读数前的验收记录](../outputs/publication_readiness_2026_09/pneuma_confirmation_rehearsal_v1/README.md)。
另可用[真实预测包演练](../outputs/publication_readiness_2026_09/pneuma_real_packet_rehearsal_v1/README.md)
在完全不读取真实未来的情况下，核对两种协议对真实数据规模的处理。

[瑞士环岛备选数据](../outputs/publication_readiness_2026_09/swiss_roundabouts_source_intake_v1/README.md)
有独立的下载、时钟分段、历史打包和匹配方案入口。每个录像或种子包可按原命令恢复。
这里只验证预测准备，尚未运行正式评分。
[瑞士评分演练](../outputs/publication_readiness_2026_09/swiss_roundabouts_readout_rehearsal_v1/README.md)
保留双车窗口的恒速回退，不把缺失的同预算对照记成零效应或安全通过。

## 3. CREATE：先查已有作业，再决定是否提交

主文这组复算在本机完成，不需重新排队。用既有获准的 CREATE 连接时，先检查
M3W 目录、配额和记录中的作业 ID。`squeue -u "$USER"` 查活动任务，
`sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed` 查终态。远程连接失败
不是作业失败，不据此取消、重启或重复提交。历史四个任务的核对结果见
[CREATE 回执](../outputs/publication_readiness_2026_09/create_accounting_recovery_20261108/README.md)，
它不证明当前有空闲 GPU。

仅对另一个确有需要、且划分和环境已冻结的 M3W 实验，在计算节点提交经核对的作业脚本。
保留 job ID、输入哈希、环境、输出目录、日志与检查点。恢复时用同一训练身份和采样状态，
先核对旧任务确实终止；`COMPLETED` 后仍需检查预测、训练步数、泄露边界和结果。
现有 [CREATE 时序实验手册](../outputs/publication_readiness_2026_09/european_temporal_auxiliary_v1/create_reproduction_zh.md)
给出已验证的项目路径、作业脚本和恢复示例，不是本轮的新提交命令。

## 4. 判断结果能支持什么

Moving 与 OpenDD 是开发证据；WEPDTOF 是冻结外部评价；Songdo 是同一采集日内的
分组校准与确认；DroneCrowd 只报告校准。三种子是同一录像的重复运行。
原 2% 是录像内风险条件，不是总体失败率；零放行不等于通过。保留原坐标与步数，
不把结果写成秒级、米级或物理安全。论文和复现包的构建通过不等于确认了部署收益，
也不等于已经提交。
