# M3W 当前论文与关键实验：中文复现入口

当前主文为 [v166](../outputs/publication_readiness_2026_09/evidence_manuscript_v166/README.md)，
配套补充材料沿用 [v165](../outputs/publication_readiness_2026_09/evidence_manuscript_v165/README.md)。
主文调整了摘要、引言和披露说明，没有改实验、图表或结论。

本地另有带证据索引的补充材料版本：保留原 50 页，新增索引和一页成对
消融。两份原 PDF 均未改动。[生成和验证说明](../outputs/publication_readiness_2026_09/supplement_evidence_guide_v1/README.md)。

一次核对主文数字、WEPDTOF 风险检查、固定预算表及最新 OpenDD 消融，
运行 `.venv-pytorch/bin/python scripts/package_m3w_reviewer_evidence.py`。
解压生成的 `reviewer_evidence.zip`，在解压目录运行
`python -I -B -S verify.py`，无需仓库或第三方依赖。
[适用范围和完整复现](../outputs/publication_readiness_2026_09/reviewer_evidence_v1/README.md)。

单独核对摘要里的核心数字可运行
`.venv-pytorch/bin/python scripts/package_m3w_main_claims.py`。
生成的本地压缩包内有 `verify.py`，解包后可离线运行，不需要仓库、
视频或模型。它只复算冻结汇总，不重跑实验。
[对应证据和命令](../outputs/publication_readiness_2026_09/main_claims_replay_v1/README.md)。

本地补充材料另有 54 页版本，末两页汇总固定预算的事后上限、重训、预算
重排及 easy 参考支持。匿名 ZIP 解包后运行 `python -I -B -S verify.py`
可复算表格和三个地点区间，不需数据、模型或仓库。
[文件、生成命令和核验结果](../outputs/publication_readiness_2026_09/budget_addendum_v1/README.md)。

WEPDTOF 还有一份更细的离线复现包：从三个风险比例重新判定通过，而不读
旧的通过标志，随后按原始来源组复算区间。解包后运行 `python verify.py`。
[构建、结果和精确 bootstrap 检查](../outputs/publication_readiness_2026_09/wepdtof_scalar_replay_v1/README.md)。

本文档从 v160 的开发相机诊断和 v159 的一次性外部检验开始。实验结果分三类：开发录像上的实测、使用未来标签的
事后上界、独立确认。前两类都不能写成第三类。DroneCrowd 的 15 个校准来源组
没有得到满足既定规则的策略，30 个独立确认来源组不用于以下命令。

## OpenDD：车辆队列上的 EqMotion 对照

先运行 `scripts/run_m3w_opendd_vehicle_eqmotion.py fit` 冻结三个种子的
42 个模型，再运行同脚本的 `score`。任一阶段中断可用原命令继续。
比较用相同队列、掩码、训练预算和 CV 偏移上限。
[命令、配对结果和画图入口](../outputs/publication_readiness_2026_09/opendd_vehicle_eqmotion_v1/README.md)。

## OpenDD：固定预算下的 easy 支持重排序

`scripts/run_m3w_opendd_support_rerank.py freeze` 复用已核验权重，先冻结
候选和校准阈值；同一脚本的 `score` 复算三个原对照并评价新规则。
[完整命令与结果](../outputs/publication_readiness_2026_09/opendd_support_rerank_v1/README.md)。
新规则未提高保守收益，不替代 CV。

## OpenDD：定位联合选择的失败

`.venv-pytorch/bin/python scripts/diagnose_m3w_opendd_pair_disagreement.py`
在原有开发集上复算三组等预算对照，拆出已观测增益、缺失步扣减和原 2%
规则下的失败转换。无新增训练或外部标签访问。
[结果与命令](../outputs/publication_readiness_2026_09/opendd_pair_disagreement_v1/README.md)。

## Songdo：从冻结候选到正式评分

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/rehearse_m3w_songdo_confirmation.py`
用合成未来覆盖两条路径：40 航次校准、160 航次确认，或 200 航次零样本
等预算效用评估。正式入口先注册已批准的来源角色；相同协议重跑时从已完成
航次接续。[两种方案、文件及命令](../outputs/publication_readiness_2026_09/songdo_protocol_paths_rehearsal_v2/README.md)。
目前无正式登记和真实 Songdo 评分。

## Songdo：评分器演练（不读真实未来）

`.venv-pytorch/bin/python scripts/verify_m3w_songdo_readout_contract.py`
使用冻结的 200 个航次候选与合成未来，核对 573 个等预算视图、缺失步扣减、
原 2% 风险规则和来源组重采样。标签坐标转换单独用合成记录测试。
[范围与命令](../outputs/publication_readiness_2026_09/songdo_readout_contract_v1/README.md)。

## Songdo：等预算四组未评分候选

运行 `.venv-pytorch/bin/python scripts/build_m3w_songdo_matched_proposals.py`，
再运行 `scripts/verify_m3w_songdo_densest_proposal.py`。191 个航次、
三个种子，四组固定同一窗口和两次等幅介入。最大窗口逐对复算一致。
这一步不读未来标签、不校准、不分正式测试组。
[比较与复现](../outputs/publication_readiness_2026_09/songdo_matched_proposals_v1/README.md)。

## Songdo：首日完整来源审计与未评分车辆预测

依次运行 `.venv-pytorch/bin/python scripts/fetch_m3w_songdo_first_day.py`、
`scripts/audit_m3w_songdo_first_day.py`、`scripts/build_m3w_songdo_first_day_vehicle_bank.py`。
20 个路口、200 个航次，逐航次保存输入与预测校验值。实际有 14,276 个不重叠
窗口、17 个候选来源组。未来坐标和正式评分都没打开；协议确定后才分配正式角色。
[步骤和核验结果](../outputs/publication_readiness_2026_09/songdo_first_day_v1/README.md)。

## OpenDD：真实车距的等容量消融

`.venv-pytorch/bin/python scripts/run_m3w_vehicle_pair_geometry.py` 依次运行
`freeze`、`score`。189 次训练比较去位置、打乱位置、真实位置，固定预算和
原 2% 规则。真实位置未改善保守误差，保留 CV。
[实验结果和命令](../outputs/publication_readiness_2026_09/opendd_vehicle_pair_geometry_v1/README.md)。

Songdo 外部准备可运行 `.venv-pytorch/bin/python scripts/audit_m3w_songdo_publisher_footprints.py`。
仅读官方目录和地图中心文件；20 个路口初筛为 17 个地图组，不评分、不分正式组。
[来源初筛说明](../outputs/publication_readiness_2026_09/songdo_publisher_footprints_v1/README.md)。

## OpenDD：直接训练两车介入结果

`.venv-pytorch/bin/python scripts/run_m3w_vehicle_pair_learning.py` 依次运行
`freeze`、`score`。126 个训练使用地点排除的预测器生成候选，两种头同标签、
同预算。直接监督找到更多过关组合，但联合头没有胜过逐车相加的对照，继续回退 CV。
另用 `.venv-pytorch/bin/python scripts/build_m3w_opendd_vehicle_observed_geometry.py`
可复建同一窗口内的车辆相对位置，未加入本次训练。
[结果和复现](../outputs/publication_readiness_2026_09/opendd_vehicle_pair_learning_v1/README.md)。

## OpenDD：等预算的两车联合选择

`.venv-pytorch/bin/python scripts/run_m3w_vehicle_pair_support.py` 依次运行
`freeze`、`score`，再运行 `scripts/diagnose_m3w_vehicle_pair_support.py`。
复用三种已有权重，不重训；三臂固定窗口和两次等幅介入。联合方案多找到
85 个过关组合，但保守配对收益为负。事后在 8,062 / 8,874 个视图中验证
有可行组合；该上限使用标签，不用来放行或调阈值。
[结果和命令](../outputs/publication_readiness_2026_09/opendd_vehicle_pair_support_v1/README.md)。

## OpenDD：用过去轨迹预测 easy 参考

`.venv-pytorch/bin/python scripts/run_m3w_vehicle_support_rank.py` 依次运行
`freeze`、`score`，可复现 63 个训练和三臂对照。每臂 9,000 次等数量、
等窗口、等半径介入。学习排序过风险和双基线的次数更多，但总收益降低；
57 次留出介入只有 18 次通过，不替换 CV。不改变原 2% 或正式测试。
[结果和命令](../outputs/publication_readiness_2026_09/opendd_vehicle_support_rank_v1/README.md)。

## OpenDD：把 easy 参考支持与风险大小拆开

用 `.venv-pytorch/bin/python scripts/run_m3w_vehicle_easy_factor.py` 依次运行
`freeze`、`score`，再运行 `scripts/diagnose_m3w_vehicle_easy_factor.py`。
本机 arm64、4 线程，231 个训练，每 100 步保存检查点。拆分模型在留出地点
放行 20 次，15 次缺少 easy 分母，其余 5 次通过。等介入次数的收益区间跨零，
不改变回退策略。未开启独立确认标签。
[结果和命令](../outputs/publication_readiness_2026_09/opendd_vehicle_easy_factor_v1/README.md)。

## OpenDD：双地点排除的车辆选择器

运行 `.venv-pytorch/bin/python scripts/run_m3w_opendd_vehicle_matched.py freeze`
生成不含目标地点和当前训练样本地点的候选，再运行 `score`。84 个训练有
每 100 步检查点。三臂均零介入，等预算收益不可估；`scripts/diagnose_m3w_vehicle_matched.py`
定位预测 easy 风险问题。事后找到的可行编辑不是模型结果，原 2% 不变。
[详细结果和命令](../outputs/publication_readiness_2026_09/opendd_vehicle_matched_v1/README.md)。

## OpenDD：车辆专用候选预测器

`.venv-pytorch/bin/python scripts/run_m3w_opendd_vehicle_residual.py` 依次运行
`extract`、`fit`、`score`。7 个留出地点、3 个种子、2 种损失，共 42 个训练；
每 100 步可恢复。带缺失步扣减的训练相对 CV 保守改善 11.54 个百分点，
但全量替换只有 14/1,500 通过原 2% 风险规则，不能部署。没打开 Songdo。
[模型、消融与复现](../outputs/publication_readiness_2026_09/opendd_vehicle_residual_v1/README.md)。

## Songdo：未评分候选轨迹与同半径搜索检查

用 `.venv-pytorch/bin/python scripts/build_m3w_songdo_candidate_bank.py` 生成候选；
再运行 `scripts/check_m3w_songdo_equal_radius_search.py`。21 个旧 OpenDD 行人
权重经过核验，只作为车辆迁移压力对照。同半径匹配在 107 车窗口上与原枚举
一致；不读取未来索引或坐标，不确定正式划分，也不报效果。
[详细复现](../outputs/publication_readiness_2026_09/songdo_candidate_bank_v1/README.md)。

## Songdo：观测输入包和来源关联

用 `.venv-pytorch/bin/python scripts/build_m3w_songdo_prefix_pack.py` 导出
1,992 个窗口、81,000 条过去轨迹。未来行号和可见掩码单独保存，不解析未来坐标。
用 `scripts/audit_m3w_songdo_prefix_linkage.py` 检查来源：A 和 B 的观测范围
重叠，前瞻性分组应把它们放在一起。未分配正式测试组或计算预测成绩。
[详细复现](../outputs/publication_readiness_2026_09/songdo_prefix_pack_v1/README.md)。

## Songdo：外部成对标注与时钟审计

按发布顺序取得最早日期的 A/B/C/E 四包。主规则下有 1,992 个非重叠的
多车窗口，1,991 个至少有一对完整未来，124 个整场完整。C 路口的漂移
敏感性检查单列，增加 458 个窗口，但不替换主结果。只读取时间和标注
存在性，不评分、不定正式划分。数据原包留在 `external_data`。
[下载、原始校验及复现命令](../outputs/publication_readiness_2026_09/songdo_traffic_intake_v1/README.md)。

## OpenDD：只在观测窗口内验证近邻滞后信息

近邻必须能在最后两个已观测步上同时超过自历史滞后和零变化预测。
4,015 个有近邻的行中 769 个通过，三套冻结预算最终在同样的 99 个对象上替换。
两种候选都没有增加满足原规则的录像种子视图，仍是 21/439。
用 `.venv-pytorch/bin/python scripts/run_m3w_opendd_peer_lag.py freeze`
冻结，再将 `freeze` 换成 `score` 评分或续跑。
[实验结果和完整命令](../outputs/publication_readiness_2026_09/opendd_peer_lag_v1/README.md)。

## OpenDD：自历史与近邻速度候选的等预算比较

两个扩展菜单每方增加 6,223 个相同半径的候选槽。按原风险和准确性要求，
存在正相对效用的录像种子视图从 21/439 增至 26/439（自历史）和 27/439（近邻）。
在完全相同的 1,038 个介入对象上直接替换时，近邻没有可靠超过自历史。
先运行 `.venv-pytorch/bin/python scripts/run_m3w_opendd_peer_transient.py freeze`，
再把 `freeze` 换成 `score`。评分保存每折进度，重复命令可续跑。
[详细结果、固定配置与复现](../outputs/publication_readiness_2026_09/opendd_peer_transient_v1/README.md)。

## OpenDD：同预算联合选择归因

两种联合头对比同一独立选择器，三方每个窗口的编辑数量及最大位移槽完全相同。
模仿头更换智能体 505 次、只换路径 188 次；直接监督头分别 165 和 130 次。
即使事后从三套预测中逐窗口选最好，保守效用最多增加 0.04417 个百分点，且
尚未施加原 2% 风险门槛。这不是可部署结果。
运行 `.venv-pytorch/bin/python scripts/audit_m3w_opendd_pair_attribution.py`。
[实验、区间与完整复现](../outputs/publication_readiness_2026_09/opendd_pair_attribution_v1/README.md)。

## P-DESTRE：等数量轨迹筛选与观测中心融合

相同 79 条保留轨迹下，按置信度、逐帧一致性、同一轨迹一致性分别保住 56、53、52 条
完整目标。后者只有一个保留子集完全可评分，原始整场并未完整。对同样 52 条完整轨迹，
仅平均两个检测器的过去坐标，OLS4 配对相对 ADE 由 28.656 降到 25.549 像素。
这是单个关联来源的定位诊断，没有评价风险或独立泛化。
[冻结、评分和等数量对照](../outputs/publication_readiness_2026_09/pdestre_track_agreement_v1/README.md)。

## P-DESTRE：相机轨迹支持与宽高字段核对

在同一 11 个日期窗口、88 帧观测上运行两种已冻结检测器。先冻结检测和在线轨迹，
再打开标注几何。按发布页写法没有稳定匹配；只交换宽高后，RAPiD 和 MobileNet
分别恢复 70/108、61/184 个具有完整标签的在线轨迹，整场完整支持仍为 0/11。
交换宽高只作为事后诊断，不用于独立结果或改门槛。
[两种字段解释、复现命令和完整标签子集结果](../outputs/publication_readiness_2026_09/pdestre_camera_support_v1/README.md)。

## P-DESTRE：真实视频可读，录像不是独立来源

对 41.5 GB 发布包按需读取，不整包下载。扫描 75 对录像和标注，只解析帧号与匿名 ID。
原帧序下有 5,113 个不重叠多智能体窗口，54,354/59,323 个目标有完整未来；
步长 12 的结构诊断为 2,191/3,686。实际解码了一段 4K 视频的前 8 帧。
75 段录像经共享 ID 连成一组，不做内部独立校准与确认，也没有跑预测评分。
[原始来源、运行命令和结果](../outputs/publication_readiness_2026_09/pdestre_source_intake_v1/README.md)。

## AV2：直接学习风险与精度的联合通过

在相同轨迹、两条编辑、三种子和训练预算下，对照直接分类与额外的跨场景排序损失。
同样选 90 个视图，原 actor-support 通过 31 个，分类通过 14 个，排序通过 15 个。
事后只在有参考的视图里比较，分别通过 33、28、34 个；排序没有证明超过原分数。
两种新模型都没有合格的校准前缀，不部署。每 50 步可恢复。
[训练、冻结、评分与复现命令](../outputs/publication_readiness_2026_09/av2_release_rank_v1/README.md)。

## AV2：区分校准迁移和排序失误

原阈值放行 17 次，仅 4 次同时通过风险和两条精度线。把留出城市每折的
介入数锁回校准值（共 9 次），仍只通过 3 次；事后同预算上限是 9 次。
即使事后知道是否有 easy 参考，按原排序也最多连续放行 10 个合格视图，
远少于 119 个合格的固定预测。事后上限不用于阈值选择或独立确认。
[复算命令、成对区间和失败分解](../outputs/publication_readiness_2026_09/av2_release_transport_v1/README.md)。

## AV2：独立支持门槛

保持同一轨迹和两条编辑不变，先按预测是否有 easy 参考做判断，再按原伤害分数排序。
0.50 门槛在同样 89 次介入中只少选 1 个缺参考的视图，实际放行集合完全不变。
0.75 敏感性对照的共同预算为 86，少选 11 个缺参考视图，但多选 11 个已观测伤害视图。
两者均不部署。原 2% 风险与 CV/OLS4 精度线不变，不是独立确认。
[冻结、评分命令和成对城市区间](../outputs/publication_readiness_2026_09/av2_separate_support_gate_v1/README.md)。

## DroneCrowd：全场景运动对照

把冻结的历史规则迁移到 MovingDroneCrowd++ 的 24 段已开放开发录像，
不使用目标域结果拟合。全部 72 个视图均未放行；固定独立轨迹全都超过全样本
2% 风险，较 OLS4 的保守增益仅 +0.079 个百分点，区间跨零。
[跨域复验、失败归因和运行命令](../outputs/publication_readiness_2026_09/moving_history_guard_transfer_v1/README.md)。

对同一独立预测单独校准相对 OLS4 收益。加上历史后，收益 MAE 降低约
0.192 个百分点；风险点估计加收益下界选中三个来源的 8 个视图，8 个均过两条
精度线和风险。仅用收益下界会误放行 2 个；同时使用两道判断也放弃了 14 个
此前通过的视图。
[独立收益检查与两道判断的复算](../outputs/publication_readiness_2026_09/dronecrowd_recording_utility_guard_v1/README.md)。

冻结轨迹不变，前四步拟合、后四步观察值回看得到的风险特征，比同规格的迁移
风险头少 1.264 个百分点 MAE（区间 0.845 至 1.723）。对数残差上界选中两个
来源的 18 个视图，均过风险、11 个同时过精度。相同视图和次数的独立轨迹
18 个都过，但仅用独立轨迹作放行决策时只选中 7 个。
[同轨迹风险对照与复现命令](../outputs/publication_readiness_2026_09/dronecrowd_recording_history_risk_v1/README.md)。

逐查询替换数不变，把风险预测按整段录像汇总。留出来源的中位校准选中 13 个
录像×种子视图，12 个过风险，4 个同时过两条精度线和风险；更保守的残差上界
没有选中任何视图。加大风险惩罚只增加风险通过数，不保留精度收益。
[同次数对照、校准与错失机会复算](../outputs/publication_readiness_2026_09/dronecrowd_recording_risk_pool_v1/README.md)。

两个模型均以 OLS4 为默认预测。逐查询匹配替换数后，直接用 OLS4 收益作监督
比 CV 监督差 0.529 个百分点；同一候选库和相同次数的事后上限为 +4.304 个百分点
（未加风险约束）。联合规则没有发生替换。三个风险头都排除了所有查询，但按实际
标签，OLS4 本身可在 4,199/11,240 个查询通过两项局部风险比值。
[成对训练、风险诊断与上限复算](../outputs/publication_readiness_2026_09/dronecrowd_ols_relative_gain_v1/README.md)。

在三个种子和逐查询相同介入数下，纯缩放比原训练差 0.551 个百分点，
缩放加噪好 0.450 个百分点，但区间跨零，且仍比 OLS4 差 0.932 个百分点。
加噪场景规则过了 88/120 个原风险检查，但精度和风险同时通过的为零；
在它的介入次数下，完美预测仍最多比 OLS4 低 50.179 个百分点。
[训练、冻结、评分及无标签复算命令](../outputs/publication_readiness_2026_09/dronecrowd_observation_augmentation_v1/README.md)。

用 14 个 MovingDroneCrowd++ 来源、三个种子训练十二步收益和伤害模型。
迁移到已开放 DroneCrowd 校准组后，仅看收益比 OLS4 差 1.384 个百分点；
场景风险预算只保留 25 次介入，即使这些预测完美，仍比 OLS4 低
至少 51.417 个百分点（组均上界）。相同次数下没有风险收益。
[训练、输入差异、冻结和上界复算](../outputs/publication_readiness_2026_09/dronecrowd_full_scene_transfer_v1/README.md)。

在原 15 个已开放校准来源组，只改 OLS4 起点不改斜率，保守精度提高
0.391 个百分点，14/15 来源组为正，但原风险和双基线只过 8/40 个录像。
同一全场景预测库的事后见证能过 40/40，不代表可部署。原两次介入协议
不变，独立确认 30 组仍封存。
[冻结、评估和无标签复算命令](../outputs/publication_readiness_2026_09/dronecrowd_motion_floor_v1/README.md)。

## Argoverse 2：完整标注下的同预算联合对照

条件伤害头在相同 90 次选择中把观测伤害从 39 降到 24，却把无参考从
6 增到 21，实际达标从 31 降到 28。原校准规则下 17 次放行仅 4 次达标。
[训练、同次数对照、概率核查及仅有参考的诊断](../outputs/publication_readiness_2026_09/av2_observed_harm_v1/README.md)。

全量训练城市的智能体用于学习 easy 支持，不改位置预测或旧收益项。
Brier 从 0.154 降到 0.117，但同次数仍是 31/90；即使事后只选有支持的
视图，也有 47 个风险失败。附本地论文图，保留原 2% 门槛。
[支持、真实伤害与缺失步的分解和复现命令](../outputs/publication_readiness_2026_09/av2_projection_support_v1/README.md)。

每场景只留一条因果选择或均匀抽取的投影，两个训练组各 4,776 条。同次数
各命中 11/90，旧成本为 31/90。已生成匿名 ZIP，能离线复算匹配计数和
46,656 个城市层配对 bootstrap 样本，不需要训练库或原始数据。
[训练、评分、匿名复算和稿件文件核对](../outputs/publication_readiness_2026_09/av2_selected_projection_risk_v1/README.md)。

用排除本来源的投影训练收益和风险打分。固定每折前五，旧成本命中 31/90，
新打分只命中 15/90；正式校准后放行 5 个，0 个达标，4 个风险失败。
[嵌套训练、来源核验、同次数对照与复现命令](../outputs/publication_readiness_2026_09/av2_projection_risk_v1/README.md)。

保持位置投影不变，只校准放行。校准选出的策略在开发留出放行 26 个视图，
只有 6 个合格，18 个未通过风险检查。旧成本打分排序较好，但放行 6 个只过 3 个。
[冻结、阈值复算、来源核验和失败诊断](../outputs/publication_readiness_2026_09/av2_position_release_v1/README.md)。

冻结的位置投影在开发留出城市通过 119 个视图，原加性参考为 89；
保守轨迹收益 +0.781 个百分点，easy 超限从 171 降到 152。相对运动损失
没有带来额外收益，两种修正都未放行。汇总脚本从逐折评分重建，不重训。
[训练、冻结、修复后的评分命令与失效归因](../outputs/publication_readiness_2026_09/av2_relative_projection_v1/README.md)。

锁定两名对象和逐步位移幅度，只训练方向。位置监督在校准角色将合格数
从 65 提高到 105，easy 超限从 184 降到 154，但相对运动误差仍变差。
方向事后上界为固定对象 216、可换对象 500，均不是可部署结果。
[方向训练、断点、评分和事后上界](../outputs/publication_readiness_2026_09/av2_projected_reference_v1/README.md)。

在校准角色重建同一预算，234 个视图存在双方都合格的事后配对，软成本参考
在其中 179 个自身不合格，真正暴露 39 个机会。历史保护把 39 增到 45，
但轨迹收益和 easy 都变差，不替换原参考。没有读取该轮留出城市标签。
[冻结、校准评分与无标签复算](../outputs/publication_readiness_2026_09/av2_causal_reference_v1/README.md)。

同一候选库上，新训逐候选回归与场景内排序，分别通过 53、48 个视图，
低于原加性成本的 89 个。精确枚举显示 317 个视图有单臂合格候选，
但双方均合格、轨迹误差不恶化且相对运动改善的视图只有 47 个。
[同条件消融、精确上界与复现命令](../outputs/publication_readiness_2026_09/av2_assignment_order_v2/README.md)。

独立约束没有解决排序：解析规则提出 47 个视图，留出开发城市放行 8 个，
只有 1 个达标；学习规则不放行。不看门槛直接按学习分数取第一，只命中
44 个合格视图，低于原加性成本的 89 个。原规则、候选库和留出角色不变。
[训练、冻结、复算和排序诊断](../outputs/publication_readiness_2026_09/av2_assignment_constraints_v1/README.md)。

相同候选库上按六城市、三种子训练分项成本：原分配、学习加性成本、再加
交互残差分别满足 63、89、80 个视图。严格联合策略在留出开发城市未放行。
事后用实际收益或风险替代估计，分别恢复到 174 和 164 个，提示下一轮优先
修收益与风险而不是加大交互模型。没有使用封存结果。
[本地训练、断点恢复、冻结评分和逐项归因](../outputs/publication_readiness_2026_09/av2_assignment_cost_v1/README.md)。

保持两条逐步位移曲线，换介入对象后，自由转向事后上限从 176 到 505；不转向、
只按历史方向搬运也有 317 个合格机会。因果联合选择只实现 72 个，且比原预测的
保守效用低 3.492 个百分点。245 次错失主要来自 easy 支持和风险预测。
[同预算换人、联合对照和失败归因](../outputs/publication_readiness_2026_09/av2_recipient_reassignment_v1/README.md)。

保持两名对象和逐步位移幅度，只训练转向。原参考 63 个合格，两个训练版本均为
62 个。逐步事后最优转向的精确上限分别为 85 个（±45 度）和 176 个（不限角度）；
原对象中仅 382/1,020 个视图有被编辑 easy 参考支持。
[训练、失败和固定几何上限复算](../outputs/publication_readiness_2026_09/av2_steered_reference_v1/README.md)。

固定原独立轨迹的两名介入对象和位移峰值，方向和时序各提供八个替换候选；
分别有 50、51 个合格机会，合并后 62 个，仍无放行。只看标签齐全的 465 个
视图，原参考也仅有 43 个合格。匿名 ZIP 可以离线复算计数及城市级区间。
[方向与时序的固定预算对照](../outputs/publication_readiness_2026_09/av2_direction_menu_v1/README.md)。

固定 935 个回放提案，分别估计原路径资格和条件替代收益。57 个原路径合格，
29 个替代满足完整对照；分解分数唯一一次留出放行失败。单独的 55 页补充材料
副本新增这一页，未改原文。[训练、冻结与生成补充页](../outputs/publication_readiness_2026_09/av2_replay_hurdle_v1/README.md)。

只改原两次介入的时间形状，保留宽候选库中 51/61 个事后机会。
历史回放选到 29 个，学习头各 24 个；带历史的头放行 5 次只过 1 次。
本地匿名 ZIP 有矢量图和离线复算，解压后运行 `python -I -B -S verify.py`。
[完整对照、训练和打包命令](../outputs/publication_readiness_2026_09/av2_incumbent_timing_v1/README.md)。

在同一合并候选库上比较匹配重训：见过时间变体后，合格逐人对照从 49
增至 59，其中有合格联合替代的视图从 40 增至 54。主方案 3 次留出放行
均失败，不部署。[训练、检查点和复现](../outputs/publication_readiness_2026_09/av2_temporal_refit_v1/README.md)。

固定两次介入及峰值，只改变位移修正的时间形状。合并候选库将相对固定逐人
对照的事后可行视图从 48 提高到 61，但候选数为原来的九倍，旧评分器选出的
合格对照从 63 降至 50，未新增可部署策略。
[候选生成、冻结和评估命令](../outputs/publication_readiness_2026_09/av2_temporal_menu_v1/README.md)。

针对实际固定逐人对照训练相对收益与风险余量。48 个事后可行的留出视图中，
二分类选中 4 个，分项损失选中 5 个，旧排序器选中 9 个。分项模型不放行；
二分类唯一放行未过原风险规则。缺标修正使连续标签有限，没有改变固定对照
的可行数。[训练、断点恢复和失败复核](../outputs/publication_readiness_2026_09/av2_relative_margins_v2/README.md)。

将 easy 参考支持和有支持时的伤害分开训练，两项覆盖分别提高
13.36 和 12.72 个百分点。分解门控放行的 3 个留出方案中
1 个单独合格，但相对同预算的固定逐人对照没有成对收益。
[条件损失、训练命令和个案核验](../outputs/publication_readiness_2026_09/av2_easy_gate_v1/README.md)。

城外排序器产生的选择用于训练独立否决器。与均匀抽候选的同架构
对照相比，总体伤害覆盖率提高 4.81 个百分点，两个 easy 项未见
明确改善，留出介入 0/3 合格。补充图和匿名离线复现包单独保存在
本地 `submission_admin`，没有改论文。
[训练、补充图与复现命令](../outputs/publication_readiness_2026_09/av2_selected_veto_v1/README.md)。

对同一批场景、同一个伤害估计器，比较固定方案和用它选出来的方案。
选优后三项覆盖率分别下降 6.77、5.57、11.17 个百分点；上尾估计
提高了强制选择的合格数，但原校准规则下仍无非空放行。
[偏乐观审计、配对区间与运行命令](../outputs/publication_readiness_2026_09/av2_harm_quantile_v1/README.md)。

将逐人增益和风险分量组合成原场景条件后，主模型 easy 支持由
382 增至 503，合格数却由 63 降至 61，留出放行只有 1/16
合格。旧的缺标枚举有 NaN 传播，修正后事后成对上限为 344，
固定对照仍为 48。[训练、逐场景核验与复现](../outputs/publication_readiness_2026_09/av2_margin_assignment_v2/README.md)。

过去可见的邻居短轨迹未改善放行。36 个场景门控拟合保留原预测和预算，
唯一留出介入不通过。事后枚举发现原固定对照只有 48 个成对可行视图，
若允许在同一候选池内改选对照，则有 334 个；后者只是上限。
[过去特征消融、逐城上限和运行命令](../outputs/publication_readiness_2026_09/av2_observability_v2/README.md)。

分别学习固定逐人对照资格与候选相对资格。18 个新拟合复用
72 个已审计的内层检查点；可行的 48 个留出视图中，直接排序选中
9 个，条件排序选中 8 个。主模型不能放行，唯一通过校准的消融
在留出城市也失败。
[训练、三组对照与失败审计](../outputs/publication_readiness_2026_09/av2_paired_qualification_v1/README.md)。

固定历史分数和逐人对照，重训 72 个内层逐人模型和 18 个代价头。
拆分已观测收益、缺标分歧和 easy 误差后，风险通过数为 158（原为 137），
但相对逐人对照的全网格成对增益仍为负。校准可行的 44 个视图只选中
1 个，无非空放行。训练命令可从检查点续跑。
[代价消融、校准上限与复现命令](../outputs/publication_readiness_2026_09/av2_forecast_cost_v2/README.md)。

固定过去 8 步、候选池和逐人对照，两组排序器仅有无短窗历史
兼容性特征不同。两臂都可合格的 48 个视图中，选中数从 0
增到 5，但完整网格比较和校准未通过。
[历史消融、缺标归因和复现命令](../outputs/publication_readiness_2026_09/av2_history_rank_v1/README.md)。

用 54 个内层 EqMotion 模型为训练城市生成未见该城市的候选，
再重训 36 个选择器头。比较组的候选和峰值预算相同。逐人对照
风险通过数增加，但加入交互项没有增益，原规则下无非空放行。
[内层训练、配对数学定义和复现命令](../outputs/publication_readiness_2026_09/av2_oof_selection_v1/README.md)。

在同一 8/12 协议上训练 EqMotion 单头，并把它作为第五类候选轨迹。
2,952 次介入的预算不变，补齐场景统一、逐人和联合三种选择。
候选上界从 300 到 334，但实际两臂合格数仍是 2。
[公开预测器训练、同预算评分和候选归因](../outputs/publication_readiness_2026_09/av2_eqmotion_v1/README.md)。

按发布方训练档案的 UUID 顺序将开发集从 128 扩到 1,024。
两个模型都在 896 个新增场景上按城市分折评分，扩容版点估计略好，
但未得到稳健放行。复现包括下载、过去窗口冻结、训练和评分。
[扩容样本、匹配对照和命令](../outputs/publication_readiness_2026_09/av2_data_scale_v1/README.md)。

新逐人对照有更多可估风险的双介入视图（32 到 45），但只有
一个同时通过风险和双基线收益。联合头用未见该训练城市的内层
对照产生监督，未实现同预算增益；原规则下没有非空放行阈值。
[重训、评分和归因命令](../outputs/publication_readiness_2026_09/av2_learned_reference_v1/README.md)。

条件收益和不合格记零两种目标各找回 3 个合格视图，但都未通过校准。
枚举表明固定逐人对照只留下 2 个两臂都能合格的视图；允许事后选取
任一合法对照后，上界是 57。后者不是已学到的策略。
[本地训练、校准和双臂上界](../outputs/publication_readiness_2026_09/av2_supported_gain_v1/README.md)。

同候选池重训 18 折，按城市分开训练、校准和评分。47 个有可行
双介入组合的评分视图中，没有一个被重训选择器选中；收益正值筛选
把大部分可行候选在校准前就排除了。复现含本地训练、评分、排序归因三条命令。
[城市隔离训练和排序归因](../outputs/publication_readiness_2026_09/av2_pair_refit_v1/README.md)。

扩展候选后，同样 27 个双槽位完整场景中，满足准确性和原风险要求的
事后可行场景从 5 增至 13。冻结模型实测仍未超过恒速基线。
[衰减加速度实验与命令](../outputs/publication_readiness_2026_09/av2_approach_menu_v1/README.md)。

从发布方训练集固定抽取六城市的 128 个场景。只用过去八步冻结预测，
再读后十二步标签。59 个完整场景中，联合选择没有可靠超过逐个选择；
双槽位的事后对照仅五个不同场景同时满足准确性和风险要求。
[实验、复现命令与汇总](../outputs/publication_readiness_2026_09/av2_native_trajectory_v1/README.md)。

## Okutama：新来源的标注节律与冻结迁移

33 个固定窗口的相机对照只使用 264 张过去图像。补偿改善了有标签的
终点误差，但完整网格的保守收益未得到支持。
[过去图像、补偿和配对复算](../outputs/publication_readiness_2026_09/okutama_past_camera_v1/README.md)。

只取训练档案标签，不取图像或测试集。排除插值框后，按十帧采样
有 91 个窗口保留至少两条完整未来。OpenDD 冻结头未超过 CV 或
同预算逐人对照的保守交互指标。
[来源、采样和离线复现](../outputs/publication_readiness_2026_09/okutama_label_histories_v1/README.md)。

## OpenDD：相同介入预算下的交互上界

保持 966 次介入和每个窗口的峰值幅度不变，限制组合相对逐人对照的
几何偏离。缺标扣减下降约 58%，但有可用组合的窗口从 150 降到 36；
直接比较和校准均未通过。两条本地命令可以分开运行，按折恢复。
[几何约束、同预算比较与复现命令](../outputs/publication_readiness_2026_09/opendd_relative_trust_v1/README.md)。

联合检索保留八个候选的预算，在评分地点找回更多合格候选，但没有证明
全网格预测提升。按同一分母拆分可见，已标注贡献接近零，缺失扣减主导。
[检索复现和缺失分解](../outputs/publication_readiness_2026_09/opendd_joint_retrieval_v1/README.md)。

用训练地点的合格候选学习相对逐个对照的交互收益，再在前八候选中排序。141 个校准机会
只有 42 个进入短名单，最终选中 22 个；评分地点未证明提升。
[条件交互收益和候选检索诊断](../outputs/publication_readiness_2026_09/opendd_conditional_pair_gain_v1/README.md)。

同窗难负例与排序损失提高了合格方案排第一的次数，但全网格交互误差
相对逐个对照变差。两种新模型都没有找到可介入的校准阈值。
[训练、同预算比较与目标审计](../outputs/publication_readiness_2026_09/opendd_pair_ranking_v1/README.md)。

把双介入作为一个决策重训 21 折，并用单独地点校准。483 个非空分数断点
均未通过；候选池有合格方案，但模型大多没有把它排到第一。
[整组准入训练和校准复算](../outputs/publication_readiness_2026_09/opendd_atomic_pair_admission_v1/README.md)。

精确求解后，固定强制槽位的 48 个录像种子视图（29 段录像）存在两种
不同的合格联合选择。加 0.25 参考门槛后，这 48 个视图没有一个保留双介入。
[约束、复算和求解命令](../outputs/publication_readiness_2026_09/opendd_distinct_supported_slates_v1/README.md)。

把衰减加速度带回 OpenDD，按七个地点、三个种子重训增益、伤害和
easy 参考预测。风险参考更完整的介入门槛降低了覆盖，但未恢复同预算联合收益。
[训练、继续运行与复算](../outputs/publication_readiness_2026_09/opendd_supported_acceleration_v1/README.md)。

在三个种子、七个地点对比三种子集标签。强制可行标签有一定信号，
但同样 282 个槽位的重排仍未改善保守交互收益。
[教师、学生和同预算复算](../outputs/publication_readiness_2026_09/opendd_subset_student_v1/README.md)。

允许退回 CV 后，保留 282 次配对介入，简单样本超限从 164 降至
3 个视图，但同预算联合方案未获合格收益。事后可行子集仍有 179 个
正收益视图。[因果放行、同预算对照和事后求解](../outputs/publication_readiness_2026_09/opendd_optional_edits_v1/README.md)。

直接学习伤害和参考分母，再按录像前缀累积，未改善原风险通过率。
保留原介入对象可以保持简单样本参考，但不能保证替换轨迹不增加误差。
[重训、风险账本和对象对照](../outputs/publication_readiness_2026_09/opendd_risk_moments_v1/README.md)。

整组交互收益打分在 21 个重训折中改善了旧联合策略，但完整网格
相对逐人方案的区间仍跨零。原 2% 规则只有 47/439 个介入视图通过；
预测缺失扣减未解决观测伤害。
[训练、复算和风险归因](../outputs/publication_readiness_2026_09/opendd_slate_relative_gain_v1/README.md)。

在相同槽位加入只用过去轨迹的转弯预测，事后可行视图从 21 增到
50。三个种子共重训 21 个折；联合策略相对逐人方案只有三个满足
全部约束的正收益视图，平均收益没有得到支持。
[冻结、重训和按地点评价](../outputs/publication_readiness_2026_09/opendd_turn_intervention_v1/README.md)。

两种菜单都保留 3,690 个固定介入槽位，按原 2% 风险规则重新验算。
完整网格的保守交互增益，事后最优由 9 增至 21 个视图；重训后的
联合选择对配对逐人方案是 0 个。
[求解、复现与按地点重采样](../outputs/publication_readiness_2026_09/opendd_expanded_interaction_frontier_v1/README.md)。

## PersonPath22：录像来源、时间轴与覆盖

官方训练和测试有三对片段来自同一原始录像、五个已知机位交叉。
首个训练录像的真实视频推理只得到一条完整匹配；用可见框也没有
恢复整组覆盖。[来源检查与本机复现](../outputs/publication_readiness_2026_09/personpath22_source_intake_v1/README.md)。

## MOT20：关联对照与运动基线

从官方档案按区间取 104 张原始帧，先生成在线轨迹再读取标签。
固定检测框后，因果运动关联将可评双人窗口从 6 增至 10（共 13 个），
但整组可评仍为零。35 条完整轨迹上，CV 和 OLS4 的 FDE 分别为
206.56 和 169.78 图像像素。[完整命令与结果](../outputs/publication_readiness_2026_09/mot20_camera_support_v1/README.md)。

## ACME：等距取窗与整组标注覆盖

两段开发录像按相机时间等距取 24 个不重叠的窗口。所有窗口均有至少
两条在线历史，但只有两个窗口有一对完整未来，整组可评为零。
23 个窗口仅从发布方参考数量就已不足，不能靠改追踪补齐原名单。
[注册、续跑、覆盖分解](../outputs/publication_readiness_2026_09/acme_camera_roster_v1/README.md)。

## OpenDD：保护所选 easy 参考

按地点排除训练 21 个支持头。预计参考下降时整对窗口回退固定方案。
联合重排可估 easy 参考从 282 增到 305 个视图，双通过从 26 到 27，
仍低于固定方案的 32；保守收益相对未加保护的重排下降 0.19142 pp。
[训练、续跑、重放](../outputs/publication_readiness_2026_09/opendd_easy_slot_support_v1/README.md)。

## OpenDD：等预算换窗口

只在同一录像的相邻非重叠窗口间重排原槽位，四组都保留 3,690 次介入。
逐人、联合双通过从各 32 降到 27、26；两种重排各丢失 17 个原有风险
通过，其中 12 个变成无法估计所选 easy 参考。这是看过两段过去窗口的
离线对照。[重放与失效归因](../outputs/publication_readiness_2026_09/opendd_paired_window_budget_v1/README.md)。

## OpenDD：扩充菜单后重训增益与伤害评分

七地点、三个种子共训练 21 个头；生成训练用神经轨迹时同时排除供体和
评估地点。固定 3,690 个介入槽位，重训联合和逐人各有 32/606 个双通过。
联合相对逐人的保守收益区间跨零。[训练、续跑和结果](../outputs/publication_readiness_2026_09/opendd_expanded_costs_v1/README.md)。

## OpenDD：加入只看自身历史的神经预测

逐人和联合都能用新路径，介入次数和每槽位移保持不变。冻结联合选择器仍为
34/606 个风险与双基线均通过的视图；加入新路径后，事后可行胜场从 9 增到
25，但不是可部署收益。[推理、同预算对照和重放](../outputs/publication_readiness_2026_09/opendd_self_menu_v1/README.md)。

## OpenDD：固定每个介入槽位的事后上限

相同的 3,690 次介入及逐槽位移，锁死单人选择后，忽略风险和两条基线
约束时，194/439 个可估视图有正优势；恢复原规则后仅剩 9 个，来自 5 段
录像和 2 个地点。这是使用未来的开发诊断，不是模型结果。
[方法、分地点结果和本机复现](../outputs/publication_readiness_2026_09/opendd_exact_slot_headroom_v2/README.md)。

## VisDrone-MOT：细切片框只补接旧轨迹

粗框优先，细框不能创建身份。单步补接和连续补接分别找回 22 和 24 条
完整轨迹，粗框对照为 21 条；未匹配历史却分别为 65、75 和 38 条。
三个方案仍没有整组可评分窗口，不采用补接规则。
[实验和离线复现命令](../outputs/publication_readiness_2026_09/visdrone_continuation_boxes_v1/README.md)。

## VisDrone-MOT：固定追踪器，只改检测切片

粗、细、合并切片分别找回 21、24、21 条完整轨迹。合并后有逐步合格框
的参考轨迹从 50 条增至 65 条，但未匹配的在线历史从 38 条增至 85 条。
原粗切片逐框复现，所有方案仍是 0/4 整组可评分窗口。
[推理、断点及离线复现命令](../outputs/publication_readiness_2026_09/visdrone_detector_tiles_v1/README.md)。

## VisDrone-MOT：固定检测框，只改在线关联

在四段已用的训练窗口，原方案、两点速度、相机补偿、两者结合分别
找回 14、11、21、14 条完整未来，四种方案均为 0/4 整组可评分窗口。
即便放弃检测框的唯一匹配，76 条完整参考轨迹中也至多 50 条在八个
过去时刻都有合格框。复现无需重新推理检测器：
[实验结果与命令](../outputs/publication_readiness_2026_09/visdrone_association_v1/README.md)。

## VisDrone-MOT：训练区标签与相机输入支持

以每四个发布帧取一步，191 个窗口有至少两条完整参考轨迹，来自 47 段录像。
固定的四窗相机试跑有 76 条完整行人参考轨迹，其中 39 条在过去八帧都能匹配
检测框，但稳定追踪并获得完整未来的仅 14 条。加入 `other people` 标签后是
17 条，整组在线轨迹都可评分的窗口仍是 0/4。没有读验证或测试数据，也没有给
任何预测器评分。[来源、逐窗统计和本机复现](../outputs/publication_readiness_2026_09/visdrone_mot_train_intake_v1/README.md)。

## KIT AIS：时间与整窗支持

只读取官方训练区；七段里只有一段按发布顺序达到 8→12，且含一次
时间间隔跳变。发布检测能找回部分轨迹，不能给整组在线轨迹评分。
[来源、计数和重放](../outputs/publication_readiness_2026_09/kit_ais_source_audit_v1/README.md)。

## OpenDD：相对运动方向消融

保留同尺寸模型、原样本和冻结预算，只比较有无方向特征。方向在
11/15 个支持组合上降低成对预测误差，但保守收益区间仍跨零，两组
都只有 33/606 个视图同时通过风险和准确性检查。
[训练、结果和复现命令](../outputs/publication_readiness_2026_09/opendd_signed_pair_geometry_v1/README.md)。

## OpenDD：直接学习两次介入的额外收益

保留冻结逐人评分与预算，只训练成对评分头。21 个本机训练组合共
6,300 步；相对逐人选择的保守收益差为 +0.00232 个百分点，地点
bootstrap 区间跨零。15 个有支持的留出地点与种子组合中，只有 1 个
预测该成对收益比直接预测零更准确。
[实验与断点恢复说明](../outputs/publication_readiness_2026_09/opendd_direct_pair_effect_v1/README.md)。

## OpenDD：只改变成对评分的同预算比较

复用 7 个地点 × 3 个种子的冻结检查点。联合与逐人对照使用完全相同的
候选轨迹、介入次数和位移预算。双人介入中有 693/1,067 个窗口改选，
但保守收益差为 -0.01003 个百分点，地点 bootstrap 区间跨零。
[结果、对照和复现命令](../outputs/publication_readiness_2026_09/opendd_slate_shared_menu_v1/README.md)。

## HT21：逐轨覆盖与整窗覆盖

用官方训练区四段录像的公开检测框生成八步历史，然后核查后十二步的
标签。逐帧时 20,064/20,702 条在线历史有完整未来，但整组在线轨迹
都可评分的多智能体窗口仅 36/285；隔 10 帧是 0/22。
[来源、分录像结果和复现命令](../outputs/publication_readiness_2026_09/ht21_detection_support_v1/README.md)。
只检查数据支持，不报告预测误差或独立安全结论。

## AerialMPT：从相机图像重算轨迹覆盖

只使用发布训练区四段够长的序列，每段固定用首个 8→12 窗口。
415 条完整参考未来只找回 3 条完整相机轨迹；换细切片无改善。
允许跨切片尺度且忽略身份的检测覆盖上界仍只有 7 条（4 像素）。
[数据位置、运行命令与逐录像结果](../outputs/publication_readiness_2026_09/aerialmpt_camera_prefix_v1/README.md)。
后续地点留出训练使用三个种子，找回 47–56 条完整未来，但没有任何
联合窗口的全部在线轨迹都可评分。阈值按校准地点的八步身份稳定性
重选后全回退。[点检测训练与阈值复现](../outputs/publication_readiness_2026_09/aerialmpt_point_detector_v1/README.md)。

## SDD：相机坐标修复与全轨迹重算

用原先冻结的相机预测，先把发布标注的参考图像像素换算到视频像素，
再把纠正后的标签对齐到全部在线轨迹。没有标签的轨迹仍留在各自窗口中。

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/audit_m3w_sdd_camera_raster.py
.venv-pytorch/bin/python scripts/audit_m3w_sdd_camera_full_roster.py
```

完整检测复现及权重位置见 [实验说明](../outputs/publication_readiness_2026_09/sdd_camera_support_v1/README.md)。
这是开发数据诊断，不是独立确认、风险校准或秒与米的换算。

## SDD：低置信度检测与轨迹标签

保持 92 个开发窗口不变，用低置信度检测补足八帧历史，不替换原有预测。
补轨后完整行人未来从 46 条增至 82 条，达到最低支持要求的录像从
1 段增至 2 段，仍只在同一场景。原无评分的 80 条轨迹中，37 条
稳定匹配骑车人或滑板者；它们没有被补计进行人指标。

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
py=.venv-pytorch/bin/python
"$py" scripts/run_m3w_sdd_camera_confidence.py freeze
"$py" scripts/run_m3w_sdd_camera_confidence.py audit
"$py" scripts/audit_m3w_sdd_camera_unknown.py
"$py" scripts/audit_m3w_sdd_camera_confidence_utility.py
```

[完整结果](../outputs/publication_readiness_2026_09/sdd_camera_support_v1/confidence_bridge_v2/README.md)。
这轮不改 2% 风险规则，也不使用预留的确认标签。

## SDD：分类别评分与联合窗口覆盖

使用同一批冻结检测结果，先锁定原有行人匹配，再给其他轨迹匹配
骑车人、滑板者标注。补轨方案额外得到 65 条完整未来，但 44 个
多智能体窗口中只有 9 个能给全部在线轨迹评分。补充类别不会改变
行人指标，也不能把可评分子集当成整场景安全证据。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/audit_m3w_sdd_camera_class_support.py
```

[分类别结果](../outputs/publication_readiness_2026_09/sdd_camera_support_v1/class_support_v2/README.md)。

## v164：学习完整介入方案

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_slate_imitation.py freeze
.venv-pytorch/bin/python scripts/run_m3w_slate_imitation.py score
.venv-pytorch/bin/python scripts/audit_m3w_slate_imitation.py
```

训练阶段只在训练录像内用未来标签找可行方案；模型输入仍只有过去和候选预测。
OpenDD 用解析候选逐地点留出；相机用既有折外神经候选训练，只评分开发相机。
`freeze` 可重跑续接检查点，`audit` 只读已有评分。
[完整对照](../outputs/publication_readiness_2026_09/source_excluded_slate_v1/README.md)。

## v163：完整轨迹的折外训练和相机选择

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_full_path_selection.py freeze
.venv-pytorch/bin/python scripts/run_m3w_full_path_selection.py score
.venv-pytorch/bin/python scripts/audit_m3w_full_path_selection.py
.venv-pytorch/bin/python scripts/run_m3w_full_path_signed_repair.py freeze
.venv-pytorch/bin/python scripts/run_m3w_full_path_signed_repair.py score
```

每个 OpenDD 来源的预测由不含该来源的模型生成。选择器用这些预测和开发标签训练，
只在已公开的 Convenience Store 相机上评分。第一轮价值头接口报错前已完成的
14 个预测器继续使用，第二版登记保留了它们的身份。运行中断后可重跑
`freeze`，从 100 步检查点继续；不需要重新训练已完成的模型。
[实验与限制](../outputs/publication_readiness_2026_09/camera_full_path_selection_v1/README.md)。

## v162：相机轨迹的配对训练和可行性检验

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_camera_noise_control.py freeze
.venv-pytorch/bin/python scripts/run_m3w_camera_noise_control.py score
.venv-pytorch/bin/python scripts/audit_m3w_camera_noise_capacity.py
```

使用 OpenDD 的 7 个开发来源训练。输入为此前冻结的三台开发相机历史。
`freeze` 可从 100 步检查点恢复，先冻结全部预测，再用 `score` 评分。
最后一条使用开发标签做事后优化，并不产生在线策略。不打开 WEPDTOF 的
13 个外部组或 DroneCrowd 确认集。详见[实验记录](../outputs/publication_readiness_2026_09/camera_noise_control_v1/README.md)。

## v160：同预算下能否追上 OLS4

```sh
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/audit_m3w_wepdtof_camera_capacity.py
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/audit_m3w_wepdtof_camera_capacity.py --replay
```

只读取 Convenience Store 的冻结相机历史及开发标签，不读取其余 13 组。
第一条从已有种子检查点续跑，第二条重新算上界并核对汇总。上界使用未来标签，
不是模型预测；过去窗口内的滚动误差排序才是因果对照。两者都不更改原风险规则。
[诊断报告](../outputs/publication_readiness_2026_09/wepdtof_camera_capacity_v1/README.md) |
[v160 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v160/README.md)。

## v159：WEPDTOF 一次性外部检验

Convenience Store 只用于开发；另外 13 个原始录像来源组的结果不用于选模型、
阈值或对照。真实输入为每个窗口 8 帧，预测之后 12 个采样步；不换算成秒或米。
沿用原 2% 规则，固定介入次数的比较只用于检验效用。空对照不可估，不计作安全通过。

在具备 v158 的登记、相机与检测器资产，以及 OpenDD 七折样条模型和训练集预测缓存后，
可按下列顺序在本机复现。`produce` 和 `train` 支持从现有检查点续跑，
`verify` 单独重新推理首窗以核对一致性。

```sh
export OPENBLAS_NUM_THREADS=1
py=.venv-pytorch/bin/python
"$py" scripts/run_m3w_wepdtof_external_camera.py produce
"$py" scripts/run_m3w_wepdtof_external_camera.py verify
"$py" scripts/run_m3w_wepdtof_external_transfer.py train
"$py" scripts/run_m3w_wepdtof_external_transfer.py pilot
"$py" scripts/run_m3w_wepdtof_external_readout.py register
"$py" scripts/run_m3w_wepdtof_external_transfer.py freeze
"$py" scripts/run_m3w_wepdtof_external_readout.py freeze
"$py" scripts/run_m3w_wepdtof_external_readout.py score
"$py" scripts/audit_m3w_wepdtof_external_failures.py
```

不要使用 `run_m3w_wepdtof_external_transfer.py score`；正式入口是
`run_m3w_wepdtof_external_readout.py score`。在已完成评分的工作区直接重跑后者，
它只重放经冻结标识核对的逐录像汇总；无需也不允许重新打开标签。
若只阅读或构建论文，不必运行评分，直接使用已提交的汇总。
[实验与结果](../outputs/publication_readiness_2026_09/wepdtof_external_transfer_v1/README.md) |
[v159 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v159/README.md)。

## v158：鱼眼相机输入与新来源

```sh
for phase in audit produce verify evaluate; do
  OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_wepdtof_source_probe.py "$phase"
done
for phase in produce verify evaluate; do
  OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_wepdtof_rapid_probe.py "$phase"
done
.venv-pytorch/bin/python scripts/audit_m3w_wepdtof_input_contrast.py
.venv-pytorch/bin/python scripts/freeze_m3w_wepdtof_external_roles.py
.venv-pytorch/bin/python scripts/freeze_m3w_wepdtof_blind_camera_plan.py
```

下载地址、发布版本和权重哈希见实验登记。RAPiD 源码固定到
`3e55294e0e419166bf152a15271c6b2fbd75b5cd`，本地需要
`pycocotools==2.0.10`。每个相机窗口单独保存，可自动续跑。
只有 Convenience Store 的几何标签被用于匹配和评分，另外 13 组只冻结角色和观察帧。
[v158 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v158/README.md)。

## v157：预测每一步的标签可用性，再决定是否偏离基线

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/train_m3w_reference_bridge.py --replay
```

去掉 `--replay` 可训练或续跑。依赖 v155 的分折、冻结预测和原训练数据；
单进程、4 个计算线程，三个对照使用相同训练抽样。阈值仅在校准来源筛选，
同位移比较只用于解释留出开发结果，不用于重选模型。
[v157 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v157/README.md)。

## v156：跨时刻分配同一个介入预算

```sh
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/audit_m3w_record_peak_allocation.py --replay
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/audit_m3w_record_peak_break_even.py
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/audit_m3w_record_peak_repair.py
```

依赖 v155 的冻结预测。第一条去掉 `--replay` 可从已有检查点续跑求解；
后两条分别估计追平 OLS4 所需的最低位移，并复核首次未判定的三个视图。
三个命令使用已开放的开发结果做事后诊断，不是在线策略或独立验证。
[v156 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v156/README.md)。

## v155：用双基线与录像风险训练初始预测

```sh
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_moving_dual_record_forecast.py --replay
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/audit_m3w_moving_initial_matched_budget.py
```

去掉 `--replay` 可训练或从检查点续跑。需要 v154 使用的 Moving 冻结候选和旧样条
检查点。每折分别使用拟合、校准和留出来源。第二条固定旧介入对象、时刻和峰值位移，
只比较达到该位移的新方向；不使用这个结果重选策略。风险规则和独立确认边界不变。
[v155 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v155/README.md)。

## v154：OpenDD 到 Moving 的决策迁移

```sh
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_cross_domain_pass_transition.py --replay
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/audit_m3w_single_exchange_envelope.py
```

去掉 `--replay` 可训练或续跑。需要 OpenDD 的冻结介入和 Moving 原拟合来源的冻结方案，
不读取 Moving A 或独立确认组。新模型只在 OpenDD 拟合。第一条固定每个查询的介入
数和峰值位移；第二条忽略风险，估算保持单查询总位移时理论上能提升多少，仅用于解释失败。
[v154 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v154/README.md)。

## v153：单查询替换的 easy 参考保护

```sh
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_opendd_counterfactual_support.py --replay
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/audit_m3w_counterfactual_support.py
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_opendd_factored_counterfactual_support.py --replay
```

去掉 `--replay` 可训练或续跑。需要 v152 的匹配候选、流式介入与检查点。局部、整段
和分解模型使用相同的介入时刻、数量和位移。中间命令是使用结果标签的事后上界，
不是在线策略或独立确认。
[v153 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v153/README.md)。

## v152：固定预算，学习整段可行的对象和形状分配

```sh
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_opendd_feasible_recipient_imitation.py --replay
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/audit_m3w_feasible_recipient_imitation.py
```

去掉 `--replay` 可训练或续跑。需要 v151 的流式数量模型、双重留出排序器和
样条检查点、冻结的外层介入。原时刻、数量、每个位置的半径不变。整段结果只用于
训练教师、事后容量与正式评分；推理时不读取结果。第二条锁定学到的对象，
检验事后改形状还剩多少容量。
[v152 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v152/README.md)。

## v151：用控制器的实际介入做校准

```sh
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_opendd_stream_spline_calibration.py --replay
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/audit_m3w_stream_spline_joint_opportunity.py
```

去掉 `--replay` 可续跑 63 个数量模型；每 100 步记录优化器检查点和心跳。
需要 v150 的生成器和选择器、双/三重留出排序器以及冻结外层路径。第一条
重现因果校准介入，保持外层对象、时刻、路径候选和半径不变。第二条只为
容量诊断使用结果标签，不能当作可部署方法或独立确认。
[v151 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v151/README.md)。

## v150：OpenDD 固定介入下的双重留出样条选择

```sh
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/finalize_m3w_opendd_nested_spline_choice.py --replay
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/audit_m3w_nested_spline_choice.py
```

去掉第一条的 `--replay` 可从优化器检查点续跑或拟合缺少的模型。需要 v149 的
本地开发数据、冻结预测和固定介入凭证；v150 不重拟合外层预测生成器。用于
校准的生成器和选择器同时排除外层地点及校准地点。各策略在相同的对象、时刻和
峰值位移下比较；只在校准地点选规则，不读取 DroneCrowd 独立确认组。
[v150 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v150/README.md)。

## v149：OpenDD 匹配训练预算与固定介入对照

```sh
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_opendd_matched_spline.py --replay
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/audit_m3w_opendd_matched_spline_slots.py
```

去掉第一条的 `--replay` 可训练或续跑。需要现有 OpenDD 开发缓存、
EqMotion 检查点和固定介入凭证。新旧模型均在六个地点拟合、留一地点，
核对相同的抽样和更新数。第二条锁定介入对象、时刻和峰值位移，
同时区分实际替换与事后有标签的候选容量。开发信号不等于部署通过。
[v149 构建](../outputs/publication_readiness_2026_09/evidence_manuscript_v149/README.md)。

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
