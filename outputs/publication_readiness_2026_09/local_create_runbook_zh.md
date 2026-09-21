# M3W 本地与 CREATE 操作记录

最近更新：2026-09-21。用途：可复现的工程与开发实验步骤，不是完整投稿实验教程。
下方带日期的历史状态只对应当时的运行；最新结果以 README_RESULTS 和对应实验报告为准。

## 当前：原生坐标损失的完整人群对照

主指标选择已授权并落实，不需要再次询问。新对照保留完整175,756条源数据索引，
以同一因果Transformer比较旧归一化损失与原生坐标损失。四个源场景逐一排除、
三个种子、共24次拟合，每次4000步。不是独立测试，也不是新部署模型。

```sh
.venv-pytorch/bin/python scripts/run_m3w_native_forecast.py --trial coupa_native_coordinate_seed17 --stop-at 100
.venv-pytorch/bin/python scripts/run_m3w_native_forecast.py --resume
.venv-pytorch/bin/python scripts/run_m3w_native_forecast.py --evaluate
.venv-pytorch/bin/python scripts/run_m3w_native_forecast.py --verify
```

首条为纳入正式预算的真实训练试跑，不看留出场景成绩。第二条从检查点继续，
并完成其他固定试验。只有24个完成收据都有效时，第三条才开始留出源场景评价。
最后一条用缓存预测重算并验证结果，不新增推理。不要同时启动两个执行器。
日志、PID心跳、检查点均在 `data/stage_cvpr2027_experiments/native_forecast_v1/`。
CPU计算线程4、inter-op1、workers0；不使用x86Conda，不调用Torch资源探测。
当前登记检查点尚未运行真实训练；实际状态以心跳结合活进程和完成收据核实。

## 已完成：成本头的恢复完整性检查

旧 ridge 入口的 `--resume` 在训练完成后会跳过 OOF 缓存和完整报告来源检查。
这一缺口已在临时合成训练中复现，不能再把旧入口的成功提示单独当成完整验证。
新入口 `scripts/train_m3w_oof_cost_head_v2.py` 保留原来的训练计算与参数，
在完成时额外绑定依赖；完成后的恢复不导入 Torch、不新增训练、不改写产物。
仅新登记作业使用新入口。旧协议绑定旧文件，不要为了迁移修改历史哈希。

本地已有六个真实历史 ridge 头，没有 v2 完成收据，应只读核查此前保留的可信
源快照，而不是事后根据当前文件补造收据：

```sh
.venv-pytorch/bin/python scripts/verify_m3w_frozen_cost_completions.py --resume
.venv-pytorch/bin/python -m pytest tests/test_m3w_cost_completion.py tests/test_m3w_cost_validation_lineage.py tests/test_m3w_experiment_contract.py -q
```

六个头的原始绑定、每头11,966条/306维和normalizer都通过。这里只读已缓存的
训练标签来校验结构与文件身份，不重算预测精度，不是新增真实训练。真实核验
无Torch导入，不需要MPS或CREATE。初次没有公开核验结果时运行同一命令去掉
`--resume`；已有结果不要删除重建。59项定向测试加2项原有跨进程恢复测试通过。

若新作业在“旧训练器完成、v2收据尚未发布”的极短窗口中断，新入口会拒绝
自动把它视为可信完成。保留所有产物，使用独立保存的绑定恢复核查；不要删
结果重跑、覆盖旧文件或手工修改收据来跳过检查。旧编排脚本仍属于冻结回放，
没有自动迁移。详细范围及限制见 `cost_completion_v2/repair_and_verification.md`。

主指标选择仍待确认，本轮不启动依赖它的新真实训练。

## 已完成：成本头训练内取证

重放12个已训练成本头，未新训练。每个头11,966条、306维特征；OOF只是轨迹
预测器没有见过目标折，成本头本身见过这些训练行，所以不能称独立验证。
全部头的总体伤害MSE优于常数，但24个原始切换资格组中23个实际净收益为负。
这把故障定位到训练内的条件成本估计，并非只在跨域后才出现。

```sh
.venv-pytorch/bin/python scripts/audit_m3w_cost_head_fit.py --family transformer --resume
.venv-pytorch/bin/python scripts/audit_m3w_cost_head_fit.py --family eqmotion --resume
.venv-pytorch/bin/python scripts/verify_m3w_cost_head_fit.py --family transformer
.venv-pytorch/bin/python scripts/verify_m3w_cost_head_fit.py --family eqmotion
.venv-pytorch/bin/python scripts/report_m3w_cost_head_fit.py
```

已完成resume只核验收据/汇总，没有新增forward。初次新目录用`--pilot`，然后
resume补齐。Transformer原CPU、EqMotion原MPS，4计算线程/1inter-op/workers0。
真正MPS推理需要获准的Metal执行环境；不静默切CPU。原批次验证每折固定取
首中末，54批共5,748条重复核验记录，特征和独立重算标签完全相同；不是全量
原始行重建。全部12个normalizer一致。45测试通过、1个默认关闭MPS测试跳过；
实际MPS模型重放单独完成。所有执行会话退出，不需要HPC。

完整报告`cost_head_fit_forensics_v1/conclusions.md`。逐行数组、checkpoint、
历史数据不上传；Git只保存代码配置和汇总。未改主指标/阈值/部署；待确认主
评价后，再登记与easy目标一致的成本学习和独立验证，不能靠训练内分数宣布成功。

## 已完成：固定决策的风险取证

本轮没有新训练或推理。读取已冻结的全部24组合、72对照，先核对原完成报告、
代码、输入和批次收据，再逐查询分析预测风险与真实已知伤害。仍使用旧主指标，
不进行阈值搜索，不打开新的测试数据，也不把旧development当独立风险校准。

```sh
.venv-pytorch/bin/python scripts/audit_m3w_frozen_risk.py --resume
.venv-pytorch/bin/python scripts/report_m3w_frozen_risk.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_frozen_risk_forensics.py tests/test_m3w_frozen_interaction.py tests/test_m3w_frozen_runtime.py tests/test_m3w_risk_calibration.py -q
```

本地已完成时，resume精确重放24个组合，增加零次计算。首次新目录可用
`--pilot-candidates 1`试跑，再resume补齐；不要删除已有结果伪装fresh。
日志/心跳/逐查询记录位于`data/stage_cvpr2027_experiments/frozen_risk_forensics_v1/`，
大文件不提交。公开报告为`frozen_risk_forensics_v1/conclusions.md`及完整表格。

全部预测预算通过，但已知标签能证明每组0至529个查询实际超预算。另一方面，
实际预算内查询仍可贡献大量easy损伤，说明“全体绝对平均伤害”并不等于
“easy相对退化<=2%”。缺失selected标签保留未知；不能填零或删除后宣称安全。
42项测试通过，独立重算核对72汇总和69,840次重复查询状态；这些不是独立样本数。
所有进程已退出，不需要HPC。仍需确认主指标，再登记对应风险目标和独立校准方案。

## 已完成：固定神经预测器的联合机制对照

本轮沿用旧协议、三个种子、两个成本头、两个固定策略，只读已经打开过的
UCY development 数据。不是重新训练，也不改变主指标。不把新增联合对照和
原来的 risk-only 对照直接比较后就声称交互贡献：需要额外的 unary-geometry
对照，保留单人几何项，只删除双人乘积项。

```sh
.venv-pytorch/bin/python scripts/run_m3w_frozen_interaction.py --family transformer --resume
.venv-pytorch/bin/python scripts/run_m3w_frozen_interaction.py --family eqmotion --resume
.venv-pytorch/bin/python scripts/analyze_m3w_frozen_interaction.py
```

已有完成标志时，`--resume` 校验输入、批次收据和汇总，增加零个推理查询，
不重写完成结果。未完成时从最后一个完整的 128-query 批次继续，不能在同一
目录并行启动第二个 runner。初次新建运行不带 `--resume`；本地已有结果不应删除重建。
Config、runner 和执行依赖有固定哈希，运行中不得修改；身份不符应诊断而不是跳过检查。

原模型对应的旧 supervised backend 与当前版本相差一个后加的模型分派分支。
runner 从指定 Git 提交恢复旧字节到 private runtime mirror，并核对原 SHA256，
不覆盖当前代码、不放宽哈希或数据边界。每一行的尺度、标签可用性和原来的
floor/candidate 预测误差必须与历史完成导出完全相同，才能进入新增比较。

本机使用 arm64 `.venv-pytorch`，CPU 4 线程、inter-op 1、workers 0。
Transformer 保留 CPU，EqMotion 保留原实验的 MPS。本轮 sandbox 下 MPS 初始化
报 macOS 版本错误，但系统实测为 arm64/macOS 15.3.1；同一命令获准在 sandbox
外执行后真实 128-query 推理通过，且误差逐行一致。这是本轮权限环境现象，
不能误写为旧的 x86_64 Conda/MKL/OpenMP 卡死，也不能偷偷改成 CPU/NumPy 再称复现。

日志、心跳、批次与完成收据保存在
`data/stage_cvpr2027_experiments/frozen_interaction_v1/<family>/`。进程存在检查
返回 permission denied 只说明无法观察，不说明卡死。用主执行会话的退出码、
最近批次时间、最终完成收据核对；`observe_m3w_frozen_progress.py --family ...
--pid ...` 仅记录心跳，不会停止或自动重启训练。无需使用 CREATE 提交重复任务。

最后的独立分析要求两个模型族都完成，核验每一行实际选择的误差和三组对比。
两段 development 录像属于同一物理场景，不能伪造 scene-bootstrap CI；三个
种子只描述拟合差异。完整 24 组合必须保留，不能选择最好的组合恢复成确认性结果。

## 已完成：补齐联合介入对照并修复数值判断

新增对照保留单人几何项，只去掉真正的双人乘积项；否则“联合优于独立”可能
只是单人分数更好。三组固定相同预测、支持、原始风险约束；后两组精确匹配
独立组在读取标签之前给出的切换数量。这不是新训练或更改主评价。

```bash
.venv-pytorch/bin/python scripts/check_m3w_interaction_controls.py --verify
.venv-pytorch/bin/python scripts/report_m3w_interaction_controls.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_interaction_controls.py tests/test_m3w_joint_intervention.py tests/test_m3w_development_evaluation.py tests/test_m3w_causal_recordings.py tests/test_m3w_sdd_step_adapter.py tests/test_m3w_sdd_auxiliary.py tests/test_m3w_source_population.py -q
```

新建结果时不加`--verify`，已存在的结果拒绝覆盖。160个构造问题、23,808种
枚举、320个最优值核验通过；33段录像的99个历史查询全部匹配，99次未来污染
不改变结果。98项相关测试通过，完成后精确语义重放通过，没有运行旧的报告
写入型全套集成测试。没有训练进程或新HPC作业。

旧求解器在hyang/video3/frame84返回成功，但比穷举最优差约4.79e-7。
等比例缩放整个目标并检查原单位原始值/对偶界/乘积关系后修复；单独设置
绝对gap=0并没有修复这个实例。旧代码与历史结果保留，没有静默换版本。
当前只有构造分数下的3个几何目标改善，未读未来误差，不能算模型提升。

科学评价变更仍待确认；不改主要指标、loss或阈值，不开启依赖新规则的训练。
后续需固定同预测器的几何独立/联合配对实验，再评价真实精度和伤害。
完整说明见`interaction_controls_v1/conclusions.md`。Stage5C/SMC关闭。

## 历史：完整源数据审计定位到评价尺度问题

这次不是新训练。四个已探索源场景共175,756个过去可用窗口，来自33段录像；
完整/部分/缺失未来标签为143,918/29,039/2,799。没有打开bookstore、主评价、
原始验证测试或外部读数。缺失未来标签仍为未知，不算成静止负例。

6,864个完整“静止后位置变化”窗口占归一化CV误差99.7481%，但原生像素误差
只占0.6660%。旧分母对静止历史使用0.001像素，明显改变了任务权重。
移动历史的七基线oracle空间15.2165%，完整总体只有0.03835%。这不是神经模型
成功，也不能把oracle当可部署结果。原始数据重放和独立误差重算已验证结论。

```bash
.venv-pytorch/bin/python scripts/audit_m3w_source_population.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_source_population.py
.venv-pytorch/bin/python scripts/report_m3w_source_population.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_source_population.py tests/test_m3w_sdd_step_adapter.py tests/test_m3w_sdd_auxiliary.py tests/test_m3w_track_event_sampling.py tests/test_m3w_causal_recordings.py -q
```

首次运行审计时不加`--verify`；完成后使用该参数精确核验，脚本拒绝覆盖已有
结果。198个数组哈希、175,756个原始索引、123万基线逐行误差、256个原始
历史标签重放和33次未来污染检查通过；47项相关测试通过。完整旧集成测试
不重跑，避免历史报告被改写。当前所需进程全部退出，无新HPC作业。

已请求确认是否另行冻结“各数据集原生单位ADE/FDE + 等场景相对基线改善”
的主评价，旧指标和负结果完整保留。确认前不改变主指标或启动依赖新指标的
训练。修改评价不是提升模型；后续仍要固定方案、匹配对照并重新验证。
详见`source_population_v1/metric_decision_pending.md`。Stage5C/SMC继续关闭。

## 历史：原始分辨率对照完成，没有可靠概率提升

固定方案 `7afef428` 已在新提取和训练前推送。首段录像实际耗时35.92秒，
1,390个原生裁剪均精确还原旧32像素RGB和覆盖计数。其余28段也已完成：
25,300个裁剪全部精确还原，累计354.50秒。95,560次运动测量和64个逻辑回归
探针均完成，分别耗时136.66秒和111.87秒。48项针对性测试通过。

较大位移标签的四种运动方案平均Brier均比对应质量对照更差，且没有任何方案
在任一留出场景超过训练集发生率基线。训练AUROC约0.79-0.80，留出场景平均
约0.48-0.49。个别排序改善不代表可靠概率或安全切换。按预先规定，不在同一
套光流特征上追加大批轨迹训练或按留出结果调阈值，没有部署升级。

```bash
.venv-pytorch/bin/python scripts/prepare_m3w_source_native_motion.py
.venv-pytorch/bin/python scripts/build_m3w_source_motion_resolution.py
.venv-pytorch/bin/python scripts/probe_m3w_source_motion_resolution.py
.venv-pytorch/bin/python scripts/build_m3w_source_motion_resolution.py --replay
.venv-pytorch/bin/python scripts/probe_m3w_source_motion_resolution.py --replay
.venv-pytorch/bin/python scripts/verify_m3w_source_motion_resolution.py
.venv-pytorch/bin/python scripts/report_m3w_source_motion_resolution.py
```

缓存和逐行预测在私有 `data/stage_cvpr2027_experiments/source_motion_resolution_v1`，
不上传Git。只有29段原有训练录像参与；bookstore、主评价、外部读数仍封存。
64个固定概率探针不是神经动力学主实验。全部对照已报告，没有挑选好看的AUROC。
运动与系数精确回放，128组评分及64组区间重算通过，128次未来标签污染检查及
48次训练角色拒绝通过；完成后再次运行零新增工作，486个产物哈希不变。
所有进程已正常退出。完整旧集成测试未重跑，因为部分测试会改写历史报告。
下一步回到完整已批准源数据的运动事件支持和现有预测器，而不是继续重复光流
设置。仍需独立校准/确认、主实验和联合介入证据；不是投稿准备完成。

## 历史：过去图像运动对照与概率探针已完成

按固定方案完成了 24 个轨迹头、24 万次更新，以及 16 个逻辑回归概率探针。
没有缩减样本或训练预算。所有 15,430 个窗口保留，23,890 对过去图像运动重新
提取并精确回放。51 个缺少局部尺度的窗口沿用原有零输出规则，没有删除。

质量信息对照和运动特征方案均未超过静止基线；运动方案的等场景 ADE 改善为
-0.000840%，属于接近零的小抖动，不是有用的动力学预测。较大位置变化的
概率排序略有改善，但 Brier 在每个留出场景都变差。不能把 AUROC 小幅改善
改写成安全门控或轨迹成功。10 像素阈值处的 11 行数值差异已单独披露。

```bash
.venv-pytorch/bin/python scripts/build_m3w_source_box_motion.py
.venv-pytorch/bin/python scripts/run_m3w_source_box_motion.py --registration configs/m3w_source_box_motion_v1.json
.venv-pytorch/bin/python scripts/run_m3w_source_box_motion.py --registration configs/m3w_source_box_motion_v1.json --replay
.venv-pytorch/bin/python scripts/analyze_m3w_source_box_motion.py --registration configs/m3w_source_box_motion_v1.json
.venv-pytorch/bin/python scripts/verify_m3w_source_box_motion.py --registration configs/m3w_source_box_motion_v1.json
.venv-pytorch/bin/python scripts/report_m3w_source_box_motion.py
.venv-pytorch/bin/python scripts/probe_m3w_source_box_motion.py --registration configs/m3w_source_box_motion_probe_v1.json
.venv-pytorch/bin/python scripts/probe_m3w_source_box_motion.py --registration configs/m3w_source_box_motion_probe_v1.json --replay
.venv-pytorch/bin/python scripts/report_m3w_source_box_motion_probe.py
```

训练 PID98675 已退出，累计拟合 656.347 秒；概率探针累计 31.422 秒。
模型、输入与系数回放通过，已完成任务恢复零新增训练。36 项针对性测试通过。
完整旧集成测试未重跑，避免改写旧阶段产物。所有当前进程已结束，不能重复
提交同一训练。私有缓存/权重/逐行预测不上传 Git。主评估与外部读数仍封存。
本轮仍是源域探索，不是确认性论文结果；下一项原始分辨率支持检查尚未运行。

## 历史：梯度诊断与输出尺度对照已完成

对上一轮24个固定模型做了训练集梯度诊断，没有新打分留出集。完整梯度和6144个
批梯度支持的是输出层集中，而不是明显的平均方向翻转。所有诊断精确重算，
已完成任务恢复零新计算，75个科学产物/检查点哈希不变。

随后固定方案5aa189b1在训练前推送，重训24个输出尺度调整模型，24万更新全部完成。
训练PID92355已正常退出；累计拟合676.292秒，包括100更新试跑。最后一层按训练
分组常数调整，不改预测边界、损失、采样或指标。日志裁剪比例降到0%，但几何/
图像相对静止基线的改善仍为-0.000251%/-0.001342%，不是预测成功。

```bash
.venv-pytorch/bin/python scripts/audit_m3w_source_gradients.py --registration configs/m3w_source_gradient_diagnostic_v1.json
.venv-pytorch/bin/python scripts/verify_m3w_source_gradients.py
.venv-pytorch/bin/python scripts/run_m3w_source_conditioned_readout.py --registration configs/m3w_source_conditioned_readout_v1.json
.venv-pytorch/bin/python scripts/run_m3w_source_conditioned_readout.py --registration configs/m3w_source_conditioned_readout_v1.json --replay
.venv-pytorch/bin/python scripts/analyze_m3w_source_conditioned_readout.py --registration configs/m3w_source_conditioned_readout_v1.json
.venv-pytorch/bin/python scripts/verify_m3w_source_conditioned_readout.py --registration configs/m3w_source_conditioned_readout_v1.json
.venv-pytorch/bin/python scripts/report_m3w_source_conditioned_readout.py --registration configs/m3w_source_conditioned_readout_v1.json
```

24个新模型的训练/留出预测精确回放，抽样与对照一致，6份跨折标签重新计算。
完成后再运行训练入口零新增更新，82个产物不变；33项针对性测试通过，未重跑
全量旧测试。大缓存、梯度数组和检查点只在本地，不上传Git。原始绑定文件不能
直接修改后沿用同一实验身份。本轮无需CREATE；既有远程认证问题未重新核验。
四个源场景结论仍仅限探索性开发，主评估/外部封存、Stage5C/SMC不变。

## 历史：重要性加权对照实验

上一轮“每个标注事件等概率”抽样改变了训练目标，造成显著退化。这轮保留相同
抽样序列，并给每个样本损失乘以 `1/(训练行数 * 抽样概率)`。不按批内权重和
再次归一化，也不裁剪权重。校正恢复的是期望损失和未裁剪梯度，不保证 Adam
更新完全相同，更不保证预测改善。所有评价行、指标、随机种子和步数固定。

配置在训练前提交为 `b2276809`，含源码与依赖哈希。真实训练使用 arm64
`.venv-pytorch`，4 个计算线程、1 个 inter-op 线程、0 个 DataLoader worker。
当前本地资源足以支持这组小预测头，不能将它的速度外推到端到端图像编码器训练。

```bash
.venv-pytorch/bin/python scripts/run_m3w_source_importance_sampling.py --registration configs/m3w_source_importance_sampling_v1.json --check-objective
.venv-pytorch/bin/python scripts/run_m3w_source_importance_sampling.py --registration configs/m3w_source_importance_sampling_v1.json
.venv-pytorch/bin/python scripts/run_m3w_source_importance_sampling.py --registration configs/m3w_source_importance_sampling_v1.json --replay
.venv-pytorch/bin/python scripts/analyze_m3w_source_importance_sampling.py --registration configs/m3w_source_importance_sampling_v1.json
.venv-pytorch/bin/python scripts/verify_m3w_source_importance_sampling.py --registration configs/m3w_source_importance_sampling_v1.json
.venv-pytorch/bin/python scripts/report_m3w_source_importance_sampling.py --registration configs/m3w_source_importance_sampling_v1.json
```

首个 100 更新试跑计入总预算，不输出预测。完整入口会恢复既有检查点；必须先
确认原进程已经终止，不能因观察超时重复启动。每 200 更新原子保存模型、优化器、
抽样 RNG、Torch RNG、权重和数据身份。已完成任务再次调用应零新增更新。
本地日志、心跳、权重和逐行预测位于 `data/stage_cvpr2027_experiments/source_importance_sampling_v1/`。
这些内容不提交 Git；只公开代码、配置、聚合结果和复现记录。

预检已完成：4 个实际训练分组的期望恒等式通过，37 项针对性测试通过，包括
小批量穷举、精确恢复、概率变化拒绝和均匀抽样下与原训练器的逐参数一致性。
完整拟合已结束：24 个新模型、24 万次更新，累计拟合 673.584 秒。24 个模型
预测精确回放，抽样与旧对照一致，恢复完成任务零新增更新、84 个文件不变。
几何和图像方案相对静止基线的改善仍为 -0.0321%/-0.2746%，是大幅减少退化，
不是新增预测优势。完整[结果与失败分析](source_importance_sampling_v1/conclusions.md)保留全部种子和场景。

四个反复探索的源场景只能支持开发结论。未来标签只用于损失或评价；供给的历史
标注可能经后续控制点插值，因此仍是 offline annotated-history，而不是实时感知。
不会打开 main/outer 结果，不启用 Stage5C/SMC，不做 metric 或 seconds-level 声明。

## 历史：SDD 状态变化支持量已全量计算，未新增训练

60 个源文件、10,616,256 行、10,300 个视频内轨迹 ID 已重新统计，并逐文件
全量重算验证。步长 1/6/12/30 仅是 raw-frame 诊断，不是正式采样选择。
步长 12 的行人完整标签有 249,384 个窗口，但固定“静止后明显移动”代理条件
只涉及 78 条轨迹、84 个不重叠片段；不重叠仍不等于统计独立。
[完整报告](sdd_state_support/conclusions.md)。

```bash
.venv-pytorch/bin/python scripts/audit_m3w_sdd_state_support.py
.venv-pytorch/bin/python scripts/audit_m3w_sdd_state_support.py --verify
MPLCONFIGDIR=/tmp/m3w-mpl XDG_CACHE_HOME=/tmp/m3w-font-cache .venv-pytorch/bin/python scripts/analyze_m3w_sdd_state_support.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_sdd_state_support.py tests/test_m3w_sdd_past_images.py tests/test_m3w_sdd_media_audit.py tests/test_m3w_sdd_image_coordinates.py tests/test_m3w_sdd_source_links.py -q
```

第一条逐视频保存收据并支持复用；第二条真的重新计算所有源统计，不覆盖原收据。
第三条区分“八个采样点中的控制点”和“过去时间范围内的全部控制点”，并输出
只含聚合计数的图。首次绘图遇到字体缓存目录不可写，使用临时缓存正常完成；
上面的可写缓存环境变量供后续运行使用，不影响科学计算。

57 项针对性测试通过；635 个真实轨迹的未来修改/截断检查不改变过去输入。
全部进程结束；全量旧测试未重跑。没有模型训练或新部署。辅助数据角色、
采样策略待登记，不自动更改主指标、封存角色或时间单位。

## 历史：SDD 过去图像读取器已验证，未新增训练

已经实现带明确缺失支持的诊断读取器，固定检查所有 60 个视频的前 64 帧。
总计 3,840 帧、79,680 标注行；部分越界 4,231 条，疑似黑边 302 条。
原始观测 RGB、去疑似黑边 RGB、两种覆盖计数和遮挡标记分别保留。
黑边推断不是人工标签，也不证明人体可见。短历史保留 mask，不用后续帧补齐。
[结果与限制](sdd_past_images/conclusions.md)。

```bash
.venv-pytorch/bin/python scripts/build_m3w_sdd_past_images.py
.venv-pytorch/bin/python scripts/build_m3w_sdd_past_images.py --verify
.venv-pytorch/bin/python -m pytest tests/test_m3w_sdd_past_images.py tests/test_m3w_sdd_source_links.py tests/test_m3w_sdd_image_coordinates.py tests/test_m3w_sdd_media_audit.py tests/test_m3w_masked_history_images.py -q
```

第一条逐视频保存完成记录和心跳，已完成部分核对 hash 后复用。修改绑定源码、
配置或输入后会拒绝旧记录，不要跳过检查。第二条独立重解码每个视频的
0/31/63 帧，并进行未来行扰动检查；不是第二次全量视频解码。
54 针对性测试通过；180 重解码帧、2,074 裁剪逐像素一致，172 个有可见 agent
的未来行扰动检查通过。恢复验证 60 条记录，零新增解码/数组写入。
全量旧测试未重跑。全部进程已结束，不需要重启或提交重复作业。

缓存约 659 MB，在忽略目录 `data/stage_cvpr2027_experiments/sdd_past_images`。
示例 API：`SDDPastImageStore(path, observation_mode='offline_annotated').inputs(query_frame=31, agent_id=known_id, length=8)`。
角色仅 `diagnostic_only`；这不是数据源训练许可。8/16/32/64 是诊断历史长度，
不能把相邻 SDD raw frames 解释成已批准的跨数据集物理时间采样。
辅助训练的数据角色和采样方案待登记，既有主任务、指标和封存角色不变。

## 历史：SDD 视频输入对应关系已诊断修复，未新增训练

完整解码60视频、522,497帧，核对10,616,256标注行，累计单视频审计346.84秒。
54视频需参考图到视频的像素尺寸映射。Nexus的10个同名视频与标注不对应；
固定字符串排序重编号假设修复后12个Nexus标注帧范围全部有视频覆盖。
不重命名/改写原文件，输出显式路径和hash。仅诊断用途，不自动加入训练。
完整[证据与限制](sdd_media_alignment/conclusions.md)。

```bash
.venv-pytorch/bin/python scripts/audit_m3w_sdd_media_alignment.py
.venv-pytorch/bin/python scripts/audit_m3w_sdd_media_alignment.py --verify-only
.venv-pytorch/bin/python scripts/audit_m3w_sdd_resize_mapping.py
.venv-pytorch/bin/python scripts/audit_m3w_sdd_nexus_identity.py
.venv-pytorch/bin/python scripts/build_m3w_sdd_diagnostic_media_links.py
.venv-pytorch/bin/python scripts/verify_m3w_sdd_media_links.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_sdd_source_links.py tests/test_m3w_sdd_image_coordinates.py tests/test_m3w_sdd_media_audit.py tests/test_m3w_masked_history_images.py -q
```

第一条逐视频保存完成记录和心跳，可恢复；完成后验证hash，不重复全量解码。
完整解码过程不加载OpenCV或Torch，避免不必要的运行库混合。
35针对性测试通过；全量旧测试未重跑。只检查当前源文件与诊断对应关系，
不是第二次全量解码或模型复现。源视频和视觉检查图不进Git。

本次源审计指出黑边、遮挡和局部可见性需要明确 mask，上方读取器完成了后续实现；
图内框仍不等于有效的人体观测，也未认证所有帧的语义对应。
坐标映射不等于米制标定，容器PTS不等于验证过的物理时间。新辅助训练角色、
采样间隔与来源支持需要另行登记；当前8到12主要任务和封存评价边界不变。

## 历史：原始分辨率与局部运动网格对照完成，结果为负

36个真实Torch模型全部完成，每个4,000更新，共144,000更新，合计拟合174.57秒。
原始96像素裁剪构建133.13秒，30,013条降采样逐一匹配旧32像素缓存。
使用全部11,966fit窗口；三种子、三个物理场景折；8观察/12预测及主要指标不变。
quality/低清池化/低清网格/原始网格相对CV改善为-1.026/-1.036/-1.081/-1.139%。
0/36通过预测改善或easy门槛。没有新部署；[完整诊断](spatial_motion/conclusions.md)。

```bash
.venv-pytorch/bin/python scripts/build_m3w_spatial_motion.py --registration configs/m3w_spatial_motion.json
.venv-pytorch/bin/python scripts/run_m3w_spatial_motion.py --registration configs/m3w_spatial_motion.json
.venv-pytorch/bin/python scripts/run_m3w_spatial_motion.py --registration configs/m3w_spatial_motion.json --replay
.venv-pytorch/bin/python scripts/audit_m3w_spatial_motion.py --registration configs/m3w_spatial_motion.json
.venv-pytorch/bin/python scripts/diagnose_m3w_spatial_motion.py --registration configs/m3w_spatial_motion.json
.venv-pytorch/bin/python scripts/verify_m3w_spatial_motion.py
.venv-pytorch/bin/python scripts/plot_m3w_spatial_motion.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_spatial_motion.py tests/test_m3w_observed_motion.py tests/test_m3w_motion_heartbeat.py tests/test_m3w_objective_alignment.py -q
```

无需安装新全局依赖；复用已登记arm64venv和隔离OpenCV/PyAV。
训练入口自动恢复未完成检查点；完成后复用已验证产物，权重与主报告不变。
36检查点逐位重放一致。独立32原始裁剪/64运动对复核一致；不是第二次全量解码。
首次构建的重复AVFoundation类警告已披露，二次解码采用不加载OpenCV的独立进程。
17针对性测试通过，全量旧测试未重跑。所有本轮进程已结束，不要重复开任务。

SDD仅做文件哈希/视频头盘点，可运行`scripts/inventory_m3w_sdd_video_headers.py`。
60文件可读不等于标注对齐或原始完整性通过；未纳入本轮训练，未验证秒/米单位。
新增辅助训练角色等待单独决定，当前协议及封存角色不变。

## 历史：过去图像运动对照完成，未修复跨场景预测

54模型全部训练完成，每个4,000更新，共216,000更新；拟合耗时合计196.69秒。
使用缓存运动特征的小MLP，不每批解码或跑CNN，速度不是删减数据所得。
全部11,966窗口保留，0/54模型超过CV，0/54通过easy门槛，不部署。
完整[结果、失败原因与边界](observed_motion_v2/conclusions.md)。

```bash
.venv-pytorch/bin/python -m pip install --only-binary=:all: --no-deps --target data/stage_cvpr2027_experiments/optical_flow_runtime opencv-python-headless==4.13.0.92
.venv-pytorch/bin/python scripts/build_m3w_observed_motion.py --registration configs/m3w_observed_motion_v2.json
.venv-pytorch/bin/python scripts/run_m3w_observed_motion_v2.py --registration configs/m3w_observed_motion_v2.json
.venv-pytorch/bin/python scripts/run_m3w_observed_motion_v2.py --registration configs/m3w_observed_motion_v2.json --replay
.venv-pytorch/bin/python scripts/diagnose_m3w_observed_motion.py --registration configs/m3w_observed_motion_v2.json
.venv-pytorch/bin/python scripts/verify_m3w_observed_motion.py
.venv-pytorch/bin/python scripts/plot_m3w_observed_motion.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_observed_motion.py tests/test_m3w_motion_heartbeat.py tests/test_m3w_objective_alignment.py -q
```

依赖只在缺失时安装，不覆盖已有版本。注册绑定本机arm64库哈希，换平台必须
登记新运行环境，不能跳过检查。源码图像、特征和权重均留在本地忽略目录。
v1因心跳日志重复pid字段失败，未得到留出预测；v2只修该问题并重新登记。
54权重重放逐位一致，完成后恢复0新增更新，权重和报告字节不变。
13针对性测试通过；全量旧测试未重跑。所有本轮任务均已退出，无等待训练。
两次独立特征提取哈希一致，仍不等于人物运动或预测准确性获得认证。

## 历史：训练目标对齐实验完成，未修复跨场景预测

同一几何网络、11,966窗口、三种子、三个物理场景；五组固定设置均完成4,000更新，
累计180,000更新，记录拟合时间174.57秒。速度来自跳过纯轨迹分支无用RGB读取，
没有缩减数据或更新数。45模型均不如CV，easy gate均失败，不部署。
训练主要指标改善最高46.40%，却不能迁移；不是仅import或空训练。

```bash
.venv-pytorch/bin/python scripts/run_m3w_objective_alignment.py --registration configs/m3w_objective_alignment.json
.venv-pytorch/bin/python scripts/run_m3w_objective_alignment.py --registration configs/m3w_objective_alignment.json --replay
.venv-pytorch/bin/python scripts/diagnose_m3w_objective_alignment.py --registration configs/m3w_objective_alignment.json
.venv-pytorch/bin/python scripts/plot_m3w_objective_alignment.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_objective_alignment.py tests/test_m3w_offline_visual_forecast.py -q
```

首条自动恢复未完成模型；完整运行已验证只复用缓存，45权重和报告字节不变。
45模型重放逐位一致，12针对性测试通过。全量历史测试未重跑。所有本轮进程已退出。
配置与源代码绑定，不直接修改后继续旧检查点。诊断使用未来标签只做误差分解，
不输入模型、不调阈值。三场景bootstrap不是独立测试证据。
完整[结果与边界](objective_alignment/conclusions.md)。

## 历史：完整 fit 数据图像对照已训练，结果为负

用户已授权选择可行路线，本轮采用标准离线标注观测。保留插值来源限制，
不再将观测定义作为阻塞；不声称严格实时传感器因果性。原8观察/12预测、
past-normalized ADE和数据角色不变。11,966个完整fit窗口全部保留，
Zara03没有视频的180个窗口带缺图mask，不删行。3个物理场景、3个种子、
4种输入对照，每个2,000次更新；不根据留出场景分数选检查点或阈值。

```bash
.venv-pytorch/bin/python scripts/run_m3w_offline_visual_forecast.py --registration configs/m3w_offline_visual_forecast.json
.venv-pytorch/bin/python scripts/analyze_m3w_offline_visual_forecast.py --registration configs/m3w_offline_visual_forecast.json
.venv-pytorch/bin/python scripts/run_m3w_offline_visual_forecast.py --registration configs/m3w_offline_visual_forecast.json --replay
.venv-pytorch/bin/python scripts/audit_m3w_offline_visual_evidence.py --registration configs/m3w_offline_visual_forecast.json
.venv-pytorch/bin/python scripts/diagnose_m3w_offline_visual_fit.py --registration configs/m3w_offline_visual_forecast.json
.venv-pytorch/bin/python scripts/diagnose_m3w_offline_visual_support.py --registration configs/m3w_offline_visual_support_diagnostic.json
```

第一条自动从完整检查点恢复未完成模型，已完成模型只验hash；不要在原进程
仍存活时重复启动。检查点保存优化器、抽样和Torch随机状态。主进程PID92582
及全部诊断进程均已正常退出，没有需等待的任务。36模型累计72,000更新、
32.48分钟记录拟合时间。36检查点预测逐位重放一致；完成后再次运行第一条，
0次新增更新，36权重和主报告字节不变。不会开启development/calibration/confirmation。
私有图像、输入、逐行预测和权重保留于被忽略的offline_visual_forecast数据目录。
本机CPU4/interop1/workers0运行；100步真实训练4.87秒，非仅import检查。

四组相对CV分别-0.58/-0.57/-0.66/-0.74%，没有升级模型。最后两条是已冻结
模型的训练集拟合与恒定特征修复诊断，不是新的独立测试或模型选择。
107项针对性测试加2项新增支持范围测试通过；未重跑无关旧全套测试。
完整结论见[结果解释](offline_visual_forecast/conclusions.md)。

## 历史：部分图像读取已实现并全量核验，训练当时待观测定义

新增masked reader保留固定裁剪位置和逐像素覆盖率，没有把缺失像素生成出来。
Zara共14,561行图像数据对应12,098个过去8步窗口，1,935个含边缘部分裁剪，均保留。
约61MB的npy memmap按源行存一份，窗口仅保存索引。所有窗口实际通过输入API读取。
旧检查的33个缺失裁剪均仍有至少52.08%可见像素；这不是预测精度提升。

完整重建9.74秒，另一次独立重建9.68秒，18个数组逐字节一致。完成缓存恢复只验证，
0次新增转换；18项针对性测试通过。无当前训练进程，未重新跑旧全套测试。

```bash
.venv-pytorch/bin/python scripts/build_m3w_zara_masked_images.py --registration configs/m3w_zara_masked_images.json --output data/stage_cvpr2027_experiments/zara_masked_images --report-dir outputs/publication_readiness_2026_09/zara_masked_images --resume
.venv-pytorch/bin/python -m pytest tests/test_m3w_masked_history_images.py tests/test_m3w_zara_past_media.py tests/test_m3w_zara_media_lineage.py -q
```

只有需要从源视频重建时，使用新的输出/报告目录并去掉resume，不覆盖现有结果。
读取必须显式指定`offline_annotation_diagnostic`或`control_as_of_query_diagnostic`。
严格控制点模式拒绝后续控制点依赖；不会偷偷删除窗口。正式训练/评价角色被拒绝，
直到观测定义得到确认。masked image修复不能自动解决离线插值的时间来源问题。

## 历史：Zara 坐标对应已修复，观测定义待确认

已逐行追溯Zara01/02共14,561行。Zara02仅用第一条精确源控制点确定原点偏移，
其余行均在0.0000054原生单位误差内还原；不是新拟合homography或米制标定。
192次过去图像请求全部解码，159个完整中心裁剪、33个边缘缺失。6张本地联系表
已检查，不能据此声称人体定位gold或独立真实时钟认证。此次没有新训练。

关键限制：97.22%/97.71%的完整过去窗口依赖查询之后的插值控制点。
“输入不直接读未来标签”与“严格实时可获得”不等价。是否保留标准离线标注任务
或另建严格source-as-of任务，已发出问题待确认；原8/12协议与主要指标不改。

以下仅在确实需要独立重跑时使用，输出目录必须不存在；已完成结果不覆盖：

```bash
.venv-pytorch/bin/python scripts/audit_m3w_zara_media_lineage.py --registration configs/m3w_zara_media_lineage.json --output data/stage_cvpr2027_experiments/zara_lineage_recheck --report-dir data/stage_cvpr2027_experiments/zara_lineage_recheck_report
.venv-pytorch/bin/python scripts/audit_m3w_zara_past_media.py --registration configs/m3w_zara_past_media.json --output data/stage_cvpr2027_experiments/zara_media_recheck --report-dir data/stage_cvpr2027_experiments/zara_media_recheck_report
.venv-pytorch/bin/python -m pytest tests/test_m3w_zara_media_lineage.py tests/test_m3w_zara_past_media.py -q
```

10项针对性测试通过；另一次真实解码复核已确认数字、私有manifest和联系表hash一致。
不重跑无关旧全套测试。进程均已退出，无等待任务。下一次训练应处理部分裁剪mask，
不要以完整图像要求静默删除入场行人。两段Zara录像仍是一个物理场景。

## 历史：相机/支持范围修复已实测，仍不部署

已完成72个固定模型处理对照，以及6个去除相机学习输入的真实重训模型。
重训每个1,000步，共6,000次更新，累计109.87秒；不是medium/full。
Hotel伤害有所减少，但ETH更差，所有结果仍未超过CV。严格训练范围回退
拒绝所有跨场景样本，0%改善不能算成功。15项针对性测试通过。

两套实验均已退出：控制session29594，训练session1153/PID67849是历史编号。
18个原模型与6个新模型预测均精确回放；再次resume只有hash验证，不重跑。
原图、缓存、逐行结果与权重保持在被忽略的本地data目录。

```bash
.venv-pytorch/bin/python scripts/run_m3w_appearance_support_control.py --registration configs/m3w_appearance_support_control.json --output data/stage_cvpr2027_experiments/appearance_support_control --report-dir outputs/publication_readiness_2026_09/appearance_support_control --resume
.venv-pytorch/bin/python scripts/run_m3w_appearance_no_camera.py --registration configs/m3w_appearance_no_camera.json --output data/stage_cvpr2027_experiments/appearance_no_camera --report-dir outputs/publication_readiness_2026_09/appearance_no_camera --resume
```

以上复用完成目录，仅校验；重新训练必须使用新的输出/报告目录，不能覆盖。
`appearance_support_control/decomposition.json`将预测变化和门控变化分开，
不是新的阈值搜索或因果证明。完整结论见`appearance_no_camera/conclusions.md`。

下一步核验已批准fit角色的Zara01/02视频与坐标对应，补训练上下文支持；
文件存在还不等于模态准入，两段录像也不是两个独立物理场景。不要继续
在这31名静止行人上反复换截断规则。原主要指标保留，新方案仍待确认。

## 历史：过去图像神经预测对照已完成，结果为负

已完成18个真实神经网络训练：2个留出训练场景、3个种子、纯几何/当前图像/
8帧过去图像，每个1,000步。合计18,000次更新，累计拟合约135.13秒，不是
medium或full。所有18个固定门控结果均未超过CV；不升级部署模型。

365个静止窗口仅对应31名行人，ETH方向训练只有5名。32个图像不完整窗口
保留并带mask。过去图像缓存不含未来标签；标签单独用于损失和评价。
ETH训练统计下78.52%的Hotel样本至少一个特征超过固定标准分数截断范围。
该现象支持分布偏移诊断，不等于已证明截断是唯一原因。

完成回放的18个检查点与保存预测完全一致。再次resume只验证18个现有模型，
不新增更新/评价，不覆盖原完成凭据。主训练进程已退出，无等待中的本轮任务。
4项针对性测试通过，未重跑不相关的旧全套测试。

复现需要本地原始视频、既有stationary cache与注册文件绑定的源版本。不要
覆盖现有输出；下面的报告目录必须是新目录，resume是验证/续接而非重新训练：

```bash
.venv-pytorch/bin/python scripts/run_m3w_past_appearance_probe.py --registration configs/m3w_past_appearance_probe.json --cache data/stage_cvpr2027_experiments/past_appearance_inputs --output data/stage_cvpr2027_experiments/past_appearance_probe --report-dir data/stage_cvpr2027_experiments/past_appearance_resume_check_2 --resume
.venv-pytorch/bin/python -m pytest tests/test_m3w_past_appearance_probe.py
```

结论见 `past_appearance_probe/conclusions.md`。零CV误差的easy样本只能报告
绝对伤害，不能把未定义的百分比当0%。后续应先检验场景/相机支持修复，
不要继续挑本轮阈值。新主要指标仍需用户确认；原8观察/12预测协议不变。

以下为各轮历史操作记录。

## 历史：移动行人的过去图像对应检查已完成

这不是神经训练或未来预测。24个fit行人，各取完整8张过去/当前图像，192次请求、
187个不同帧。固定24像素搜索范围对Hotel过小，56/84对真实观察位移在范围外。
登记后只扩大至64像素，Hotel同样本匹配改善，ETH反而受干扰变差。两个版本均保留，
边缘缺失不当作零误差，未选择最佳范围用于预测。原生帧对应有局部支持，但没有
证明身体朝向、物理时间或全局标注准确。下一步须登记图像特征对照，不能把匹配当预测。

```bash
.venv-pytorch/bin/python scripts/audit_m3w_past_motion_correspondence.py --registration configs/m3w_past_motion_correspondence.json --output data/stage_cvpr2027_experiments/past_motion_correspondence_recheck --report-dir data/stage_cvpr2027_experiments/past_motion_correspondence_recheck_reports
.venv-pytorch/bin/python scripts/audit_m3w_past_motion_correspondence.py --registration configs/m3w_past_motion_search_support.json --output data/stage_cvpr2027_experiments/past_motion_search_support_recheck --report-dir data/stage_cvpr2027_experiments/past_motion_search_support_recheck_reports
.venv-pytorch/bin/python scripts/summarize_m3w_past_motion_correspondence.py --report-dir data/stage_cvpr2027_experiments/past_motion_comparison_recheck
.venv-pytorch/bin/python -m pytest tests/test_m3w_past_motion_correspondence.py tests/test_m3w_past_motion_summary.py tests/test_m3w_past_video_alignment.py -q
```

前两条只有需要重解码时才运行，目录必须是新目录。第三条核验并比较原始已完成的
两次运行，不会自动改读recheck目录，也不是新的模型拟合。原报告被后续登记绑定，
不能覆盖。14项测试通过，完整旧测试未重跑；本步骤无活动进程。所有原图、联系表
和逐行记录仅留被忽略的本地data目录。两次处理约6.90/12.59秒，不是长时间训练。

## 当前：过去视频读取与投影轴检查已完成

真实解码成功，不是新模型训练。31个fit行人的62次首个过去帧/当前帧请求全部取得，
共48个不同帧；58个矩形未被画面边缘裁剪，但不代表人体完整覆盖。Hotel按上游
行列约定投影后画面内比例为99.83%，旧xy解释为82.78%。原预测和旧报告不改写。
ETH时间冲突、人物定位及过去帧对应仍未独立认证，不能直接开放正式视觉训练。

原执行因NumPy整数不能序列化而失败，保留原目录及source snapshot。成功版本是v2。
PyAV18.1.0仅安装于被忽略的media_decode_runtime，未改PyTorch环境。已有18项相关
测试通过，未重跑完整旧测试集。当前本步骤无活动进程。

以下仅用于需要重新解码时的独立复核；目录必须不存在，禁止覆盖绑定的原始产物：

```bash
.venv-pytorch/bin/python scripts/audit_m3w_past_video_alignment.py --registration configs/m3w_past_video_alignment_v2.json --output data/stage_cvpr2027_experiments/past_video_alignment_recheck --report-dir data/stage_cvpr2027_experiments/past_video_alignment_recheck_reports
.venv-pytorch/bin/python scripts/render_m3w_video_audit_contacts.py --source data/stage_cvpr2027_experiments/past_video_alignment_recheck
.venv-pytorch/bin/python -m pytest tests/test_m3w_past_video_alignment.py tests/test_m3w_stationary_scene_context.py tests/test_m3w_stationary_label_resolution.py -q
```

渲染图及逐行记录只留本地。原v1登记绑定修复前代码，不应改hash来伪装成功。
先核对标注点语义、时序对应和遮挡，再登记过去RGB方向预测对照；不能把读取成功
当成视觉贡献或将原生步数换算为已验证秒数。八观察/十二预测及待确认的主指标决定不变。

## 当前：条件均值与几何中位数对照已完成

不是新神经网络训练。18个固定树模型设置保持原特征、标签、划分和0.9门限，只改
条件点预测。ETH全部归零，Hotel伤害降低但仍为负。没有任何设置超过CV，不能部署。
原始23个未达数值容差的点单独修复22个，剩余1个明确保留approximate标记。
23项定向测试通过；全部18个原始计算记录hash恢复核验通过。当前无活动进程。

```bash
.venv-pytorch/bin/python scripts/run_m3w_conditional_ade_probe.py --registration configs/m3w_conditional_ade_probe.json --output data/stage_cvpr2027_experiments/conditional_ade_probe --report-dir data/stage_cvpr2027_experiments/conditional_ade_probe_resume_reports --resume
.venv-pytorch/bin/python scripts/refine_m3w_conditional_ade_probe.py --registration configs/m3w_conditional_ade_refinement.json --output data/stage_cvpr2027_experiments/conditional_ade_refinement_recheck --report-dir data/stage_cvpr2027_experiments/conditional_ade_refinement_recheck_reports
.venv-pytorch/bin/python -m pytest tests/test_m3w_conditional_geometric_median.py tests/test_m3w_geometric_median_refinement.py tests/test_m3w_stationary_scene_context.py tests/test_m3w_stationary_start_probe.py -q
```

首次运行需新目录并去掉resume。复算目录存在时再用新名字，不能覆盖被后续登记
绑定的报告。resume只核验复用，不是重新训练。剩余近收敛点不能改为已认证；数值
修复也不能选择正结果。所有权重、轨迹支持、逐行输出仍仅留本地。公开结论见
`conditional_ade_probe_refined/conclusions.md`。原主指标不变，历史暴露状态不变。

## 当前：标签精度与隐藏匀速假说检验已完成

本轮没有重新训练神经网络。365行fit-only诊断显示，文本取整和指定半像素区间内的
隐藏匀速运动都不能解释大部分静止历史预测误差。ETH/Hotel无法由后者解释的行占
相应误差98.30%/97.21%。这不等于证明人体真实启动或标注无误。原失败模型仍失败。
72个已保存修正版模型核验回放，24项定向测试通过，当前无活动任务。

以下复算使用新目录，不覆盖被后续登记绑定的原始报告。它们只是只读来源的标签侧
诊断，不产生可部署特征；所有未来标签、可行性标记只用于检查，不得进入推理。

```bash
.venv-pytorch/bin/python scripts/audit_m3w_stationary_label_resolution.py --registration configs/m3w_stationary_label_resolution.json --source data/stage_cvpr2027_experiments/stationary_scene_probe_v2 --output data/stage_cvpr2027_experiments/stationary_label_resolution_recheck --report-dir data/stage_cvpr2027_experiments/stationary_label_resolution_recheck_reports
.venv-pytorch/bin/python scripts/audit_m3w_quantized_cv_feasibility.py --registration configs/m3w_quantized_cv_feasibility.json --output data/stage_cvpr2027_experiments/quantized_cv_feasibility_recheck --report-dir data/stage_cvpr2027_experiments/quantized_cv_feasibility_recheck_reports
.venv-pytorch/bin/python -m pytest tests/test_m3w_quantized_motion_feasibility.py tests/test_m3w_stationary_label_resolution.py tests/test_m3w_stationary_scene_context.py tests/test_m3w_stationary_start_probe.py -q
```

已存在复算目录时不要删除旧证据，改用新的目录名。新主指标协议仍待用户决定。
结论在 `stationary_label_resolution/conclusions.md`。原始数据、逐行结果、模型
不能提交Git。下方保留历史实验的复现方式，不代表这些实验仍在运行。

## 当前：静态场景与起步方向实验已完成

原始和角点修正版各72个分类/轨迹回归模型均已完成，不是新Transformer训练。
局部起步概率改善没有变成轨迹改善；修正后36个轨迹回归器及固定保护门全部没有
正收益。已有模型不升级。结果见 `stationary_scene_probe_v2/conclusions.md`。
14项定向检查通过；144个保存模型回放一致。当前无本轮活动任务。

```bash
.venv-pytorch/bin/python scripts/run_m3w_stationary_scene_probe.py --registration configs/m3w_stationary_scene_probe_v2.json --source data/stage_cvpr2027_experiments/stationary_start_probe_v2 --output data/stage_cvpr2027_experiments/stationary_scene_probe_v2 --report-dir data/stage_cvpr2027_experiments/stationary_scene_probe_v2/resume_reports --resume
.venv-pytorch/bin/python scripts/summarize_m3w_stationary_scene_probe.py --original data/stage_cvpr2027_experiments/stationary_scene_probe --repaired data/stage_cvpr2027_experiments/stationary_scene_probe_v2 --report-dir outputs/publication_readiness_2026_09/stationary_scene_probe_v2
.venv-pytorch/bin/python -m pytest tests/test_m3w_stationary_scene_context.py tests/test_m3w_stationary_start_probe.py -q
```

resume使用单独报告目录，保留原始fresh训练记录；首次运行去掉resume并指定新的
输出目录。旧版本代码留在其本地 `source_snapshot`，不能修改旧hash来加载新版。
两版特征/标签的native来源和全部365行保持一致；仅重复角点的参考方向判定修复。
14项测试覆盖未来字段拒绝、几何变换、角点处理、零基线easy伤害以及实验表完整性。
参考图包含人且时间不明，不作为输入；静态XML只作为未经验证的障碍代理使用。
新主指标协议仍待用户决定，不能把这里的分类分数写成正式预测成功。

## 当前：静止历史与邻居上下文检验已完成

这一轮只用冻结的 fit 数据，完成原始文件核对、24 个有序特征分类器、24 个低维
汇总特征分类器，以及全部48个保存模型的预测回放。没有训练新轨迹预测器，也未
打开 development/calibration/confirmation 标签。当前无本轮活动任务。

365个窗口实际是31个agent、45个静止片段。低维几何特征树模型只在 Hotel->ETH
方向改善概率误差，反向未通过；不能升级部署或写成轨迹改善。结论见
`stationary_start_probe/conclusions.md`。7项定向检查通过，不是全套测试重跑。

```bash
.venv-pytorch/bin/python scripts/run_m3w_stationary_start_probe.py --registration configs/m3w_stationary_start_probe_v2.json --output data/stage_cvpr2027_experiments/stationary_start_probe_v2 --report-dir data/stage_cvpr2027_experiments/stationary_probe_resume_reports --resume
.venv-pytorch/bin/python scripts/run_m3w_stationary_pooled_probe.py --registration configs/m3w_stationary_pooled_probe.json --source data/stage_cvpr2027_experiments/stationary_start_probe_v2 --output data/stage_cvpr2027_experiments/stationary_pooled_probe --report data/stage_cvpr2027_experiments/stationary_pooled_probe/resume_report.json --resume
.venv-pytorch/bin/python scripts/summarize_m3w_stationary_probe.py --source data/stage_cvpr2027_experiments/stationary_start_probe_v2 --pooled data/stage_cvpr2027_experiments/stationary_pooled_probe --report-dir outputs/publication_readiness_2026_09/stationary_start_probe
.venv-pytorch/bin/python -m pytest tests/test_m3w_stationary_start_probe.py -q
```

已有模型核验复用标记 `cached_verified`，不能写成重新训练。上面专门将 resume
报告写到另一路径，因为后续登记绑定了原始 `metrics.json`，不能覆盖其 fresh_run
训练记录。首次没有缓存时需去掉 `--resume`，并按登记和原始报告路径依次运行。
最初序列化失败的登记与源码快照保留作溯源，不应修改旧hash后声称复现成功。

## 已完成：v7 成本敏感对照

24 个新选择头各训练 1,000 次；原预测器不重训。两个预测器家族、三个种子、
线性/64宽 MLP、成本上限1/10全部完成。当前无活动训练进程。不要把下方历史
“待运行”的对照说明当成最新状态。结果在 `8to12_deferral_v7/results.md`。

```bash
.venv-pytorch/bin/python scripts/run_m3w_registered_deferral.py --registration configs/m3w_deferral_v7_supplement.json --output data/stage_cvpr2027_experiments/8to12_deferral_v7 --resume
.venv-pytorch/bin/python scripts/summarize_m3w_registered_deferral.py --study data/stage_cvpr2027_experiments/8to12_deferral_v7 --report-dir outputs/publication_readiness_2026_09/8to12_deferral_v7
.venv-pytorch/bin/python -m pytest tests/test_m3w_registered_deferral.py tests/test_m3w_cost_sensitive_deferral.py tests/test_m3w_deferral_development.py tests/test_m3w_deferral_summary.py -q
```

新的补充登记独立绑定父协议摘要和补充源码，不改旧协议/源码 hash。首次输出目录
尚不存在时去掉 `--resume`；完成后按当前 receipt 核验复用，不能写成 fresh 训练。
若源码或缓存身份变更，应保留旧快照并登记新版本，不覆盖旧 hash。checkpoint、
OOF缓存、逐行结果只留本地；Git仅包含代码、登记、报告和轻量聚合指标。

全部24个设置均未同时满足正改善和 easy<=2%。42项训练/接入回归和8项报告检查
通过，不代表模型通过研究gate；也未重新运行完整legacy测试套件。该对照没有
M3W的风险预算，不能假设相同介入率。一个开发物理场景不足以计算独立场景CI。

## 当前 v7：固定数据下的残差参数化对照

两版本的三种子完整训练、主评价与 raw50/同人数补充均已结束。主 runner 和补充
进程均正常退出，不需要重新启动。轻量结论在 `8to12_residual_pair_v7/conclusions.md`
及 `supplement_conclusions.md`。幅度约束减少漂移，但安全选择后改善极小，联合
选择仍无额外收益，不能写成新部署或投稿达标。24 项最终报告/回放检查通过。

以下命令保留作同源码版本的复现入口；已有完整任务会按身份核验复用。数据缓存、
权重、完整逐行导出均留本地。核验完成应看 completion receipt 与报告 hash，
不能仅凭最后一条周期 heartbeat 判定任务仍运行。

新协议仅比较 CV 初始化残差与相同残差的过去运动量幅度约束，保留 v6 数据和评价。
两种结构先完成真实 100 步 CPU4 试跑，再从同一检查点继续完成三种子预算：

```bash
.venv-pytorch/bin/python scripts/run_m3w_residual_parameterization_pair.py
```

运行前先检查 `8to12_residual_skip_v7` / `8to12_motion_bounded_v7` 的实际 child PID；
健康任务不要重复启动。训练中不要编辑冻结源码和协议。完整预算结束后运行：

```bash
.venv-pytorch/bin/python scripts/compare_m3w_residual_parameterizations.py --skip outputs/publication_readiness_2026_09/8to12_residual_skip_v7/metrics.json --bounded outputs/publication_readiness_2026_09/8to12_motion_bounded_v7/metrics.json --reference-v6 outputs/publication_readiness_2026_09/8to12_transformer_v6/metrics.json --output-dir outputs/publication_readiness_2026_09/8to12_residual_pair_v7
```

v7 下方的 v6 复现命令应使用原源码快照 `052bcc64`，不能在新工厂实现上改旧协议
hash 强行加载。新 v7 协议摘要为 `f82ec96eaaea9ecd7ab7218829f99f43e4c91ab1c7af3ef27d38f08fd0f1621a`。
新补充评价要显式传入 `--decision outputs/publication_readiness_2026_09/forecast_supplement_v7_decision.md`，
不能误用默认的 v6 决定。所有新输出仍为 development-only，幅度限制不等于安全保证。

完整补充入口示例（第二版本将 `residual_skip` 换成 `motion_bounded`）：

```bash
.venv-pytorch/bin/python scripts/evaluate_m3w_forecast_supplement.py --protocol configs/m3w_8to12_residual_parameterization_v7.json --study-dir data/stage_cvpr2027_experiments/8to12_residual_skip_v7 --output-dir data/stage_cvpr2027_experiments/8to12_residual_skip_v7_supplement --device cpu --threads 4 --decision outputs/publication_readiness_2026_09/forecast_supplement_v7_decision.md --resume
.venv-pytorch/bin/python scripts/summarize_m3w_forecast_supplement.py --cache-dir data/stage_cvpr2027_experiments/8to12_residual_skip_v7_supplement --report-dir outputs/publication_readiness_2026_09/8to12_residual_skip_v7_supplement
.venv-pytorch/bin/python scripts/audit_m3w_supplement_replay.py --study-dir data/stage_cvpr2027_experiments/8to12_residual_skip_v7 --supplement-dir data/stage_cvpr2027_experiments/8to12_residual_skip_v7_supplement --report-dir outputs/publication_readiness_2026_09/8to12_residual_skip_v7_supplement
```

两版本已完成逐行误差/决策回放核验，不要把 completed resume 记作一次 fresh 训练。

## 当前 v6：输入数值范围修复后的完整匹配训练

当前两个模型族已经完成三种子、24 个 forecaster 和 6 个 neural cost head 的全部
预算与主评价，六次开发选择均为 CV。不要为了更新状态重复训练。下方旧阶段中的
“正在运行”均为历史快照，以本节和当前产物的 completion/heartbeat 为准。

v5 seed29 在 CPU/MPS 都出现 nonfinite loss，不能跳过种子写完整研究。
v6 保留数据、目标和评价尺度，只在网络内部按 past-only 共同尺度变换再还原。
配置已冻结；不要在运行中改绑定代码、配置或决定文件。

```bash
.venv-pytorch/bin/python scripts/run_m3w_conditioned_predictor_pair.py
```

此命令完整执行两个模型、三种子、每种子 full + 三个物理场景留出模型，各 10,000
updates，再做 OOF 风险头和开发评价。输出目录为 `8to12_eqmotion_v6` 与
`8to12_transformer_v6`，位于忽略的 `data/stage_cvpr2027_experiments` 下。
启动前先检查 runner heartbeat 对应的进程是否仍运行，禁止重复启动。
已完成文件按 hash 核验复用，未完成从同一 checkpoint 恢复；不会自动降为 quick。
seed29 的 100 步数值试跑会继续到完整预算，不是另一个准确率结果。

全部完成后：

```bash
.venv-pytorch/bin/python scripts/compare_m3w_public_predictors.py --transformer outputs/publication_readiness_2026_09/8to12_transformer_v6/metrics.json --eqmotion outputs/publication_readiness_2026_09/8to12_eqmotion_v6/metrics.json --output-dir outputs/publication_readiness_2026_09/8to12_public_predictors_v6
```

### 已完成的固定预测补充

两模型的 raw-frame t+50 和同实际切换人数对照都已完成。补充目录的最后一条周期
heartbeat 仍写 running，最终状态应核对 `completion.json` 的报告 hash 及 PID
是否已退出，不能只看旧 heartbeat。所有原始预测误差和普通策略决策与主评价逐行
相同。结果为负或无稳定收益，不因此修改主指标或模型选择。

相同源码/协议/设备下恢复或核验已有 EqMotion 补充：

```bash
.venv-pytorch/bin/python scripts/evaluate_m3w_forecast_supplement.py --protocol configs/m3w_8to12_conditioned_context_v6.json --study-dir data/stage_cvpr2027_experiments/8to12_eqmotion_v6 --output-dir data/stage_cvpr2027_experiments/8to12_eqmotion_v6_supplement --device mps --threads 4 --resume
.venv-pytorch/bin/python scripts/summarize_m3w_forecast_supplement.py --cache-dir data/stage_cvpr2027_experiments/8to12_eqmotion_v6_supplement --report-dir outputs/publication_readiness_2026_09/8to12_eqmotion_v6_supplement
.venv-pytorch/bin/python scripts/audit_m3w_supplement_replay.py --study-dir data/stage_cvpr2027_experiments/8to12_eqmotion_v6 --supplement-dir data/stage_cvpr2027_experiments/8to12_eqmotion_v6_supplement --report-dir outputs/publication_readiness_2026_09/8to12_eqmotion_v6_supplement
```

Transformer 使用相同命令结构，将目录中的 `eqmotion` 换成 `transformer`，设备改为
`--device cpu`。首次在新目录构建不加 `--resume`；已完成目录不要删除或强行覆盖。
恢复核验旧缓存是 cached_verified，不是新的训练结果。row cache、checkpoint 不进 Git。
raw50 是同一 12 步预测的原生精确 5 步前缀，不是重新训练/重新条件化的 t50 模型。
一个物理开发场景不能给出有效跨场景 CI，三种子标准差也不能替代它。

仍是 development-only，不是独立确认或公开 best-of-20 复现。出现 nonfinite 时保存
错误日志/权重身份，不丢弃失败 fold，不静默换设备。以下旧命令保留用于各自源码版本
追溯，不能在 v6 目录/新源码上修改旧 hash 强行恢复。

## 历史 v5：连续身份上下文修复后的匹配训练

当前源码快照 `5e0f7be9`，协议 `configs/m3w_8to12_continuous_context_v5.json`。
它保留既有 fit/fold/seed/metric，只把 Students01 的 20 点片段包装替换为本地连续
轨迹源，保留原始身份、精度、短轨迹和尾段。历史暴露不改变，仍是 development-only。
v4 第一个 full forecaster 已完成 10,000 updates；runner 因上游 context 可用性问题
主动停止在开发评价前，不是卡死。不要把该单个完成的 fit 写成完整三种子实验。

```bash
.venv-pytorch/bin/python scripts/train_m3w_causal_forecaster.py --protocol configs/m3w_8to12_continuous_context_v5.json --preflight-only
.venv-pytorch/bin/python scripts/run_m3w_continuous_predictor_pair.py
```

第二条顺序执行 EqMotion K=1 的三种子 full/fold/OOF/开发评价，再执行匹配 Transformer。
EqMotion 使用显式 MPS，Transformer 使用 CPU4，interop1、workers0。不存在静默设备回退；
已完成文件核验复用，未完成检查点恢复。不要重复启动同一 runner。
两者都使用完整对齐的过去邻居、batch32、Smooth-L1、lr0.0003、10,000 updates。
输出位于 `data/stage_cvpr2027_experiments/8to12_eqmotion_v5` 和
`8to12_transformer_v5`；逐子任务 `.log`、`heartbeat.jsonl`、`latest.pt` 与根目录
`runner_heartbeat.json` 保留 PID/step/时间。当前正在运行，尚无完整 v5 精度结果。

如果 MPS 确实运行失败，可停止同一任务后用显式 CPU 命令恢复，不改模型/预算：

```bash
.venv-pytorch/bin/python scripts/run_m3w_8to12_development.py --protocol configs/m3w_8to12_continuous_context_v5.json --config configs/m3w_eqmotion_8to12_pruned_v4.json --study-dir data/stage_cvpr2027_experiments/8to12_eqmotion_v5 --device cpu --threads 4
```

跨设备恢复须报告 runtime history；不能冒充始终在 MPS 运行。优先在相同设备恢复。
全部种子完成后才能汇总并配对比较；不能缺 seed、截短预算或换支持集合：

```bash
.venv-pytorch/bin/python scripts/compare_m3w_public_predictors.py --transformer outputs/publication_readiness_2026_09/8to12_transformer_v5/metrics.json --eqmotion outputs/publication_readiness_2026_09/8to12_eqmotion_v5/metrics.json --output-dir outputs/publication_readiness_2026_09/8to12_public_predictors_v5
```

源审计和新缓存准备脚本是首次构建工具；已有版本拒绝覆盖。不要在训练中改其绑定文件。
Git 不含原始数据、作者核心源码、缓存或权重，独立机器还需核验合法来源及相同 hash。
当前开发协议不是已验证的完整匿名复现包。原始连续注释是否有未来依赖的插值仍未完全
审计；修复包装可用性不等于证明 sensor-as-of 因果性、米或秒。

固定完整窗口 benchmark 与完整场景因果介入的目标不同。本项目发现的是后者要求下的
上下文选择问题，不据此宣称所有使用完整窗口的公开基准无效或作者有不当行为。

## 已完成的真实 8→12 开发实验与稳健损失对照

用户已选定观察 8 步、预测 12 步为主任务，raw-frame t+50 为补充；其余路线按研究授权确定。
使用新协议 `configs/m3w_8to12_development_v1.json`，不修改下文保留的旧独立实验草案。
这是 development-only：既有历史暴露不清零，没有 calibration/confirmation 角色或安全保证。

v1 MSE 实验的可复现代码版本为 `707d4017`。下面四条 v1 命令须使用该版本；
新实现的 source hash 不同会被正确拒绝，不能修改旧协议 hash 绕过检查。
请在独立工作目录检出该版本，不回退或覆盖当前有未提交工作的目录。
该目录还需有按原相对路径放置、hash 一致且获准使用的本地数据与缓存；
Git 不包含这些文件。跨 workspace 的数据软链接会被路径边界拒绝。
这里验证的是本机训练与 checkpoint 恢复，尚未完成独立机器的匿名复现打包。

```bash
.venv-pytorch/bin/python scripts/train_m3w_causal_forecaster.py --protocol configs/m3w_8to12_development_v1.json --preflight-only
.venv-pytorch/bin/python scripts/run_m3w_8to12_development.py --device cpu --threads 4
.venv-pytorch/bin/python scripts/summarize_m3w_8to12_development.py
.venv-pytorch/bin/python scripts/audit_m3w_8to12_training_scale.py
```

第二条依次执行 17/29/43 三个种子，各自完整预测器、三个场景留出预测器、相同 OOF 数据上的
ridge/神经 gain-harm head 和开发评价。已完成模型核验后复用，未完成 checkpoint 原样恢复；
逐 recording 评价也支持恢复。不要同时启动两个相同 runner，不要运行中修改协议绑定源码。
只运行特定登记种子可显式加 `--seeds 17`，但不能写成三种子完成。

输出位于本地忽略目录 `data/stage_cvpr2027_experiments/8to12_v1/`，包含每个子任务的
`.log`、checkpoint、artifact manifest 和 `heartbeat.jsonl`。根目录 `runner_heartbeat.json`
记录父/子 PID、当前命令、耗时；有进度而慢不是卡死。训练中断从最后完整 checkpoint 恢复。
汇总脚本必须等全部登记种子评价完成再运行，只导出轻量 JSON/Markdown。

v1 已完成全部 15 个神经 fit，每个 1,000 updates；三种子均选择 CV floor，不是成功提升。
首个完整模型曾由 50-step checkpoint 恢复到 1,000 updates。三个种子不是三个独立地点；
当前开发侧只有 University 一个物理场景，不能据此给出跨场景 bootstrap CI。
旧草案默认 preflight 仍会拒绝，这是预期行为；必须显式传入新的开发协议。
本地数据、OOF 缓存和 checkpoint 不提交 Git。CREATE 连接条件未改善，本轮不提交远程作业。

稳健损失 v2 只改 forecaster MSE 为 Smooth-L1(beta=1)，不覆盖 v1。下面须在其源码
快照 `9e592089` 及匹配本地缓存下运行，不是当前 v5 源码；不能改 hash 强行兼容：

```bash
.venv-pytorch/bin/python scripts/run_m3w_8to12_development.py --protocol configs/m3w_8to12_robust_v2.json --config configs/m3w_intervention_robust_backend.json --study-dir data/stage_cvpr2027_experiments/8to12_robust_v2 --device cpu --threads 4
.venv-pytorch/bin/python scripts/summarize_m3w_8to12_development.py --protocol configs/m3w_8to12_robust_v2.json --study-dir data/stage_cvpr2027_experiments/8to12_robust_v2 --report-dir outputs/publication_readiness_2026_09/8to12_robust_v2
```

重新运行相同 runner 会核验并恢复，不能同时开两个。prepare 脚本只用于首次登记新版本，
已有版本拒绝覆盖。不同损失的训练 loss 数值不能互相比较；只比较不变的开发评价。
v2 现已三种子全部完成，均选择 floor；完整对照在
[paired report](8to12_robust_v2/robust_loss_comparison.md)。v2 自身已完成；当前 v5 另在运行。

## 已验证的环境

Apple Silicon 上使用仓库的 `.venv-pytorch/bin/python`，不要使用默认 x86_64 Conda。新的运行探针会在导入 Torch 前拒绝 macOS x86_64；历史问题是架构/运行库组合，不是“PyTorch 不能多线程”。

计算线程和数据读取进程分开设置：CPU 计算线程分别试 4、8；interop=1；DataLoader workers=0。不启动额外 DataLoader 子进程。当前 Torch 构建为 arm64、Accelerate BLAS、USE_MKL=OFF；以实际训练而非 import 作为验证。

## 因果数据重建

在仓库根目录运行：

```bash
.venv-pytorch/bin/python scripts/build_m3w_causal_recordings.py
.venv-pytorch/bin/python scripts/audit_m3w_causal_recordings.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_causal_recordings.py
```

原始第三方轨迹只读；新数据写入被 Git 忽略的 `data/stage_cvpr2027_causal/`。原始文件哈希和目录分组不符会停止。报告写入 `outputs/publication_readiness_2026_09/`。

`RecordingWindows.get_scene_inputs(frame_id, horizon_raw)` 根据当前可见性和过去历史选择 agents，不看未来是否完整；`get_scene_labels(...)` 另行返回未来标签与 loss mask。新接口可支持后续联合介入策略，不等于已完成联合策略训练。

最初的 8 观测/12 预测只是可用性视图；现已成为顶部版本化开发协议的主任务，仍不等于独立正式测试。raw-frame t25 没有精确窗口，不能用邻近帧补成“t25”。所有视图可能重叠，不是独立统计样本。

## 外部原始来源审计

```bash
.venv-pytorch/bin/python scripts/audit_m3w_external_sources.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_external_source_audit.py -q
.venv-pytorch/bin/python scripts/check_m3w_citr_source_manifest.py --report outputs/publication_readiness_2026_09/external_source_audit/citr_upstream_identity_system_tls.json
```

第一条只读取本地 OpenTraj 中 GC/HERMES/Wild-Track/CITR/VRU 原始标注，重算来源 hash、轨迹长度、断帧和精确标签可用性，报告写到 `external_source_audit/`。它不训练、不插值、不生成大缓存，也不设定正式 split。报告会覆盖同名审计快照；需要保留旧快照时指定新的 `--report-dir`。第三条读取 CITR 作者公开仓库的 commit/tree 元数据并核对本地 raw CSV 的 Git blob hash，不下载轨迹。本轮 344/344 匹配；其 README 的 340 行人数与 raw 实际 318 的差异仍需解释。脚本使用系统 curl 的正常 TLS 校验，未禁用证书验证。

GC 的 raw stride=20，因此不能用插值补出 raw t50 来冒充真实标签。VRU 的 measurement ID 不等于已验证的全局视频 frame；两条异常时钟轨迹被隔离，不能把每个对象的时间零点拼成同场邻居。`obs8/pred12` 只是逐源观测步可用性，不自动代表相同物理时长或正式批准的主协议。十五项测试验证这些边界，来源权限、历史暴露及独立场景资格仍须另行核实。

## CITR 诊断缓存接入

```bash
.venv-pytorch/bin/python scripts/build_m3w_citr_recordings.py
.venv-pytorch/bin/python scripts/build_m3w_citr_recordings.py --resume --report outputs/publication_readiness_2026_09/citr_causal_intake/resume_report.json
.venv-pytorch/bin/python -m pytest tests/test_m3w_citr_recordings.py tests/test_m3w_causal_recordings.py -q
```

第一条仅在新输出目录执行；已有缓存不覆盖。缓存位于忽略目录 `data/stage_cvpr2027_causal/citr_diagnostic/`。中断后加 `--resume`：已完成 clip 核验后复用，带本次 ownership 标记的未完成 clip 重建；原始 CSV/代码/缓存身份改变即拒绝。报告应另取文件名保留 fresh 与 cached_verified 的区分；逐 clip 写 PID 心跳。

实际转换 38 clips / 95,648 点 / 318 行人轨迹 / 26 车辆轨迹，逐行映射回原 CSV。raw10/25/50/100 视图分别 89,800 / 84,640 / 76,040 / 58,840；obs8/pred12 为 89,112，均含两类 agent 且可能重叠。`CITRRecordingWindows` 提供因果场景输入与独立标签 API，不读取 filtered 速度。agent type 目前为元数据，不等于已训练类型 embedding。

全体 clip 属于同一物理地点，全部 `diagnostic_only`，未进入正式协议。来源条件、历史暴露和独立确认资格尚未解决；没有新增预测训练/精度/metric/seconds 声明。详见 [接入与边界](citr_causal_intake/implementation_and_limits.md)。

## DUT 原始标注诊断接入

```bash
.venv-pytorch/bin/python scripts/fetch_m3w_dut_annotations.py --download
.venv-pytorch/bin/python scripts/build_m3w_dut_recordings.py
.venv-pytorch/bin/python scripts/build_m3w_dut_recordings.py --resume --report outputs/publication_readiness_2026_09/dut_causal_intake/resume_report.json
.venv-pytorch/bin/python scripts/audit_m3w_dut_intake.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_dut_recordings.py -q
```

下载器固定作者 commit，白名单只含 raw CSV、比例尺及只读说明/审计代码，检查普通文件/Git blob/TLS/总大小，已有内容不符就拒绝覆盖。默认没有 `--download` 只读远程目录。不要运行下载的第三方 filter script；其初始速度使用未来帧。原始文件写到忽略目录 `external_data/DUT_author_raw/`，缓存写到 `data/stage_cvpr2027_causal/dut_diagnostic/`。首次转换需新目录，后续显式 `--resume`，逐 clip 记录 PID 心跳；另取 `--report` 保留 fresh / cached_verified 历史。

实际 28 clips / 457,686 原始行，独立重数与逐行来源一致；全体只属于两个物理地点。`dut_intersection_04` 两个行人 ID 在相同 145 帧上坐标完全相同，报告标为质量隔离待审，缓存保留原样，不能直接纳入正式实验。README 的 meter 概括与作者对 raw 除比例尺的代码存在语义冲突，因此不应用比例尺、不新增 metric/seconds 声明。全部 diagnostic_only，未获 source-use 或正式 split 批准，无训练/预测精度。详见 [报告](dut_causal_intake/implementation_and_limits.md)。

## 正式实验入口与拒绝行为

诊断来源现在有额外准入层，详见 [实现与边界](intake_admission/implementation_and_limits.md)。带 `source_conditions_review` 的缓存不能仅靠协议 `approved` 字段进入 fit/development/calibration/confirmation：需要 hash-bound `intake_screen`，以及另行审查的 `source_use_decision`（reviewer、decision reference、允许用途、证据文件哈希）。screen 中摘要写“无问题”不能覆盖底层 audit 的 quarantine。原始数据不得静默去重，登记 excluded 不代表获准训练。

```bash
.venv-pytorch/bin/python -m pytest tests/test_m3w_intake_admission.py tests/test_m3w_experiment_contract.py -q
.venv-pytorch/bin/python scripts/build_m3w_dut_admission_screen.py --output outputs/publication_readiness_2026_09/intake_admission/new_screen.json --report outputs/publication_readiness_2026_09/intake_admission/new_refusals.json
```

第二条只验证旧源/缓存并生成准入证据，不签发 permission、不分配真实角色、不调用 future label API；已有输出不覆盖。DUT 28 段 × 4 种用途的检查全部拒绝：4 项质量隔离，108 项 source-use 未批准。正式草案未改；后续科学决定明确后，要连同新 checker 版本审查绑定，不能改旧 approval hash 来绕过。旧 canonical declarations 和绕过 `ExperimentContract` 的 legacy 脚本没有被这一补丁升级为全局授权系统。

```bash
.venv-pytorch/bin/python scripts/check_m3w_experiment_contract.py --report-dir outputs/publication_readiness_2026_09/experiment_contract/unapproved_rejection
.venv-pytorch/bin/python -m pytest tests/test_m3w_experiment_contract.py -q
```

当前第一条应返回 exit 2，因为协议仍未批准；这是安全拒绝，不是运行时卡死。第二条是 25 项合成/临时目录测试，不代表真实实验获批。不要通过把草案 `status` 改成 approved 或清空历史暴露来绕过：源数据、主评价规则、独立 calibration/confirmation 和用户决策仍须核实。

后续新训练代码需实际通过 `ExperimentContract` 打开对应用途的 recording；OOF gain/harm 使用的所有父模型也必须通过整 fold 来源检查。校准和最终测试前分别登记固定候选集合及 hash，恢复时身份必须相同；完成后不再作为 fresh confirmation 重跑。当前旧训练器尚未全部接入，不能宣称全仓库防泄露已经完成。详见 [实现与限制](experiment_contract/implementation_and_limits.md)。

## 新预测训练与 OOF 代价入口

新入口从同一份 hash-bound 协议打开因果 recording，不读取旧 teacher cache。当前可直接复现的是工程检查：

```bash
.venv-pytorch/bin/python scripts/check_m3w_supervised_inputs.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_supervised_intervention.py -q
.venv-pytorch/bin/python scripts/train_m3w_causal_forecaster.py --preflight-only
.venv-pytorch/bin/python scripts/train_m3w_oof_cost_head.py --preflight-only
```

第一条使用随机权重检查真实输入，**不计算预测准确性**；本轮 24 queries / 345 agents、未来标签调用 0 次，另 3 个 ETH-eth 查询无 raw50 过去网格支持。第二条 17 项测试只在临时合成 fixture 上训练；与来源、协议、joint/reader 相关检查合计 102 passed。最后两条对当前真实草案返回 exit 2 是预期结果，不要改写批准字段来绕过它。

正式开跑前要有经过确认的主时域、角色/分组、指标和风险预算；`configs/m3w_intervention_backend.json` 只有模型和训练技术参数，不批准任何科学规则。查看实际必需参数：

```bash
.venv-pytorch/bin/python scripts/train_m3w_causal_forecaster.py --help
.venv-pytorch/bin/python scripts/train_m3w_oof_cost_head.py --help
```

预测训练要求显式传入协议、fit recordings、baseline、seed 和新的 output directory。`--device cpu` 或 `--device mps` 显式选择设备，`--threads` 默认 4，interop=1，workers=0；不会把 CPU fallback 伪装成 MPS。当前是固定预算 fit-only，不是 validation-selected best。loss 为 masked normalized coordinate MSE，不等于论文 ADE/FDE。

恢复使用相同参数、相同 output directory 加 `--resume`。`latest.pt` 含参数、optimizer、采样顺序/游标、采样 RNG、Torch RNG、loss 和 runtime 段；身份或训练设置变化会拒绝恢复。中断从最后完整 checkpoint 恢复，不保存半次 optimizer update。已完成的相同运行返回 `cached_verified`，不继续偷偷更新权重。保存和心跳频率由技术 config 控制；测试用 `--stop-after` 可模拟分段训练。CPU/MPS 合成恢复数值一致不代表任意设备相同或已证明 12 小时稳定性。

OOF cost 入口还需要显式 `--fold-models` 映射和 producer artifact manifests。它会检查整个 held-out fit fold 对所有父模型的暴露，再读取监督；不允许只检查当前单个 recording。完整 fold 输出有哈希与 run identity，`--resume` 复用已核验 fold，未完成 fold 重算，篡改缓存会拒绝。每 batch 写进度心跳。线性 cost control 的 mean/scale 仅在 fit OOF 特征上拟合；验证、校准、最终测试还没有执行。大缓存、模型、optimizer 文件不进入 Git。

更完整的实现和边界见 [supervised backend](supervised_backend/implementation_and_limits.md)。旧的 runtime probe 与下面的 joint checks 仍是独立工程证据，不应与新真实预测实验混为一谈。

## 神经 Gain/Harm Head

新入口 `scripts/train_m3w_neural_cost_head.py` 从上一节 ridge OOF 运行目录读取完整折缓存，核验 hash/身份后训练小型连续代价回归 head，不重抽数据、不训练预测器本身。需显式 `--protocol`、`--artifacts`、`--oof-cache-dir`、`--seed`、`--output-dir`；协议中须绑定 `gain_harm_training` 的 width、loss=`squared_benefit_harm` 和 fit_settings（steps/batch_size/learning_rate/checkpoint_every/heartbeat_every）。未替真实实验选择这些值。

```bash
.venv-pytorch/bin/python scripts/train_m3w_neural_cost_head.py --preflight-only
.venv-pytorch/bin/python scripts/train_m3w_neural_cost_head.py --help
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 .venv-pytorch/bin/python -m pytest tests/test_m3w_neural_gain_harm.py -q
```

第一条当前应 exit 2，不导入 Torch 训练。恢复用 `--resume`，可用 `--stop-after` 演练中断；只有完整预算结束才产生 `artifact.json`。产物可作为 development plan 的 risk head，与 ridge 共用 forecaster、baseline、policy grid；输入 fingerprint 相同是配对前提，不应只比较模型名称。输出是 expected benefit/harm，不是安全概率。显式 MPS 测试及权限边界、未标注样本切换的负结果见 [说明](neural_cost_head/implementation_and_limits.md)。运行中不要修改绑定源码；旧实验不靠改 hash 强行恢复。

## 开发集选模与五种控制策略

```bash
.venv-pytorch/bin/python -m pytest tests/test_m3w_development_evaluation.py -q
.venv-pytorch/bin/python scripts/check_m3w_development_inputs.py
.venv-pytorch/bin/python scripts/evaluate_m3w_development.py --preflight-only
.venv-pytorch/bin/python scripts/evaluate_m3w_development.py --help
```

17 项新检查包括临时合成端到端训练/OOF/开发比较、不同采样网格、缺失未来标签、easy guard、按物理场景 bootstrap 和缓存篡改拒绝。与先前相关检查合计 119 passed。真实输入 probe 使用随机神经权重和常数合成代价，不读真实标签；23 queries / 274 agents 的五种控制在未来位置破坏后保持不变。不要把它当作外部准确性结果。当前真实 draft 的 preflight 仍返回 exit 2。

正式评价需要已批准协议中的 `development_evaluation` 规则，以及显式 `--plan` 和完整 `--artifacts`。规则包括 error unit、label coverage、query stride、每个 recording 的图/距离代理、easy/hard 定义、可选 arms、bootstrap seed 和 policy grid。easy 定义必须与协议风险定义相同。候选 plan 引用实际 forecaster/risk artifact、baseline 和有哈希的 cost report；不能以旧 test-exposed teacher 充数。

入口只打开 development role。先把整个 candidate family、代码、配置、产物身份写入 `run_identity.json`，再读开发标签；每完成一个 candidate/recording 原子写缓存和 receipt。`--resume` 复用已核验结果，未完成的 recording 重算。已完成的相同运行返回 `cached_verified`，换权重/阈值/缓存拒绝。`heartbeat.jsonl` 记录 PID 和场景处理进度。使用新输出目录开始独立开发实验，不覆盖旧研究结果。

选模结果写入 `selected_policy.json` 并记录 development 暴露与父产物；没有合格候选就保留 floor，且 `deployment_approved=false`。这里的 matched 指同预测和同预算上限，不是自动相同介入率或已校准风险。汇总选择目前只支持明确批准的 past-normalized error，dataset-local 原始 ADE/FDE 按 recording 单列；不同未验证坐标不能直接池化。独立风险校准和最终确认还未执行。详见 [实现与限制](development_evaluation/implementation_and_limits.md)。

## 相同实际介入数量的诊断对照

```bash
.venv-pytorch/bin/python scripts/check_m3w_matched_coverage.py --report-dir outputs/publication_readiness_2026_09/matched_coverage/final_version
.venv-pytorch/bin/python -m pytest tests/test_m3w_matched_coverage.py tests/test_m3w_joint_intervention.py -q
```

第一条会覆盖所指定目录的 `checks.json`，保存新轮次时使用新的 report directory。脚本执行合成 MILP/穷举对照，以及 canonical/CITR 真实过去输入检查；使用随机模型和常数风险分数，不读取真实未来标签，不训练、不计算 accuracy、不批准正式协议。

`compare_at_independent_coverage` 先确定 independent 的实际数量 k，再在同一支持集和 predicted-harm cap 内固定 joint 的数量。`decide_scene(..., include_matched_coverage=True)` 显式启用额外诊断分支，默认五控制不变。matched branch 可能在正数量约束下选择预计收益较差的组合，**不得作为部署替代**。求解失败须保留 unmatched，0-count 须单列；完整标签子集的切换率也须另报，不能从全体 agent 数量一致推断它一致。

最终版 80 个合成问题全部与穷举一致；真实输入 61 queries / 618 agents 的未来破坏不改变决策。新增 20 项测试，合并相关回归 224 passed。一次在测试中修改代码导致恢复身份校验拒绝，已固定源码完整重跑；运行实际训练/恢复期间不要修改其绑定模块。这里没有新增真实预测改善或部署。详见 [方法与边界](matched_coverage/method_and_limits.md)。

## Cost-sensitive deferral 文献对照

```bash
.venv-pytorch/bin/python scripts/train_m3w_deferral_control.py --preflight-only
.venv-pytorch/bin/python scripts/train_m3w_deferral_control.py --help
.venv-pytorch/bin/python scripts/check_m3w_deferral_control.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_cost_sensitive_deferral.py -q
```

真实草案的 preflight 应返回 exit 2，不启动训练。第三条只训练构造代价的线性 gate，不能当真实轨迹效果；会写指定目录的检查报告，保留旧轮次时用新的 `--report-dir`。本轮验证 CPU，不新增 MPS 或长时稳定性声明。

正式入口需明确 `--protocol`、`--workspace-root`、`--artifacts`、`--fold-models`、`--baseline`、`--seed`、`--output-dir`。批准协议中还必须绑定 `comparators.cost_sensitive_deferral` 的 `cost_bound`、`width`、`fit_settings`，不从 test 估计范围。`width=0` 是 linear；其他为小 MLP。fit_settings 包含 steps/batch_size/learning_rate/checkpoint_every/heartbeat_every。CLI 的 `--batch-size` 仅用于折外特征提取。当前没有替用户选择真实 bound 或训练设置。

恢复加 `--resume`；`--stop-after-folds` 和 `--stop-after` 分别可用于折完成/optimizer step 中断演练。折缓存 hash、模型身份、上游暴露和 fit-only 标准化受检查。`head/latest.pt` 保存参数/optimizer/RNG/cursor，心跳带 PID；只从完整 checkpoint 恢复。完整产物保存 `artifact.json`，部分 checkpoint 不能当完成模型加载。缓存、checkpoint 不提交 Git。

这个对照保留两个预测的连续误差，不是 oracle winner 分类。输出 logit margin 不是 gain 或 harm probability，不能直接接进安全阈值冒充校准。正式 development/calibration/confirmation arms 尚未改变；需批准后按同预测器、同数据用途和匹配覆盖率规则接入。详见 [方法与局限](deferral_control/method_and_limits.md)。

## Deferral 的统一开发评价

```bash
.venv-pytorch/bin/python -m pytest tests/test_m3w_deferral_development.py -q
.venv-pytorch/bin/python scripts/evaluate_m3w_development.py --preflight-only
```

第一条覆盖真实生产 CLI 的合成训练/比较/恢复，不读取真实最终测试。第二条当前应 exit 2，未获批准不能启动。完整合成链是 3 个 forecasters 各 8 updates、64 条匹配 OOF 输入、ridge 和 8-update deferral、27 agent queries / 16 scene queries；单物理场景不生成 CI。这个短训练中 deferral 比 floor 差，不能当文献方法的正式实验结果。

批准协议可显式增加 `development_evaluation.diagnostic_controls: ["cost_sensitive_deferral"]`；每个 plan candidate 同时提供 deferral_head_id/deferral_report_path/deferral_report_sha256。不能只传一个新权重然后默认参与评价。两种 cost-head CLI 都须导出相同 `oof_feature_identity`；旧报告缺少摘要时会拒绝，不准手工伪造摘要恢复成“已匹配”。head seed、OOF/final predictor seed 和 architecture 必须一致。整组产物校验完成后才读开发标签。

六臂共享一次 forecast，但 deferral 不强加 M3W risk budget。结果单列 unconstrained 和事后 reference-budget 检查，配对差值/CI 不意味着已匹配实际 coverage 或 risk。仍只允许原有 guarded arms 进入自动选择；calibration/confirmation 的新 arm 尚未注册。恢复命令与既有 development CLI 一致；source/plan/产物/缓存改变会拒绝。相关回归 258 passed，15 个新用例，详见 [接入与边界](deferral_development/implementation_and_limits.md)。

## EqMotion 官方核心适配检查

```bash
.venv-pytorch/bin/python scripts/fetch_m3w_eqmotion_source.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_eqmotion_adapter.py -q
.venv-pytorch/bin/python scripts/check_m3w_eqmotion_inputs.py --device cpu
.venv-pytorch/bin/python scripts/check_m3w_eqmotion_inputs.py --device mps
M3W_EQMOTION_TEST_DEVICE=mps .venv-pytorch/bin/python -m pytest tests/test_m3w_eqmotion_adapter.py::test_cross_process_checkpoint_resume -q
.venv-pytorch/bin/python scripts/train_m3w_causal_forecaster.py --config configs/m3w_eqmotion_fixed_head.json --preflight-only
```

下载器只获取固定作者 commit 的 8 个代码/文档文件并逐字节核对 Git blob；不下载数据和预训练权重，不执行作者 trainer/preprocessor。保留 MIT 文件；第三方内容在 `data/stage_cvpr2027_causal/third_party/`，不进入 Git。源码缺失时 pytest 明确 skip 可选集成项，不应称为验证成功；先执行下载器再检查。正式训练前仍须批准科学协议，最后一条当前应返回 exit 2。

这里是固定单头 K=1 适配，不是论文的 minADE20/minFDE20。8/12 只用于适配工程与提案，不替用户确认正式时域。不同固定网格需不同模型，不自动插值。邻居只按过去完整度/时间对齐纳入；请报告这一支持差异，正式比较增加相同邻居支持的控制组。未来标签不参与输入或选择哪一个头。

MPS 检查需主机 Metal 访问权限。若沙箱返回 macOS 版本不支持，先核对真实系统和权限，不改用 x86 Conda；本次 macOS 15.3.1 沙箱外成功。MPS 合成跨进程恢复 3+5 对连续 8 updates 的参数、loss、optimizer 完全一致。恢复修复：AdamW 的非 capturable step 保持 CPU，其余状态由 `load_state_dict` 按参数设备处理；不再手工把所有 state 搬到 MPS。代码 hash 变化后，旧 checkpoint 按旧版本保留，不能改元数据绕过恢复身份检查。

两种设备真实输入 probe 各检查 27 queries / 330 agents，使用随机权重，未来标签调用 0，不是预测准确性或长训练稳定性结果。报告位于 [public_baselines](public_baselines/compatibility_and_limits.md)。技术配置的 10,000 steps 尚未运行；不能把测试的 8-step 合成训练写成完整公开基线训练。

## 冻结策略的场景级风险校准

```bash
.venv-pytorch/bin/python scripts/audit_m3w_risk_support.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_risk_calibration.py -q
.venv-pytorch/bin/python scripts/calibrate_m3w_intervention.py --preflight-only
.venv-pytorch/bin/python scripts/calibrate_m3w_intervention.py --help
```

第一条仅核验现有 metadata/array 身份、计算示例统计界与合成相关性诊断，不读真实未来标签。当前真实 draft 的 preflight 应返回 exit 2，不修改 approval 使其通过。新增 21 项用例与相关回归共 157 passed；没有真实校准或独立确认结果。

正式入口需要批准的 `calibration_evaluation`：error unit、标签规则、采样步长、各录像局部几何、场景内聚合方式，以及 `policy_priority`。风险必须显式定义为有界 clipped positive excess 或 harm event，指定 clipping scale/margin、tolerance、delta。不能把示例值当用户选择，也不能把 0.02 的 clipped risk 写成 2% easy relative error 保证。

传入 `--protocol`、全部上游 `--artifacts`、冻结顺序的 `--policy-ids`、新的 `--output-dir`，以及显式 `--device cpu|mps`、`--threads 4`。只接受实际 development export 的选中策略；report、plan、实现、模型、cost 和递归来源都要一致。读取校准标签前先锁定候选，后续只能筛选不能重拟合或调阈值。缺失标签的介入按最坏有界损失处理；没有介入只保证相对 baseline 零差异，不保证其预测准确。

每完成一组 policy/recording 写原子 rows cache 和 hash receipt，逐场景写 PID/progress 心跳。中断后原参数加 `--resume`；已完成部分复用，未完成 recording 重算。完成后的同一任务只做 identity verification，返回 `cached_verified`。身份或缓存变化拒绝继续；不能删 claim 然后把新模型当同一次校准。结果、calibrator artifact、completion receipt 保留在输出目录，实际大缓存不提交。该接口不打开 confirmation role，也不意味着确认评估已实现或完成。

当前仅六个物理场景组且均有开发暴露，没有批准的独立校准用途。实现中的 Hoeffding/union bound 需要独立同分布场景假设，分组命名本身不证明该假设。详见 [校准实现和限制](risk_calibration/implementation_and_limits.md) 与 [支持审计](risk_calibration/support_audit.md)。下一步不能以扩充重叠窗口替代独立场景。

## 冻结最终比较与三个训练种子

```bash
.venv-pytorch/bin/python -m pytest tests/test_m3w_confirmation_evaluation.py -q
.venv-pytorch/bin/python scripts/evaluate_m3w_confirmation.py --preflight-only
.venv-pytorch/bin/python scripts/evaluate_m3w_confirmation.py --help
```

当前真实草案预检仍应返回 exit 2，不得手动补 approval 来启动。新增 20 项用例及相关回归 177 passed，验证的是合成真实 Torch 训练/OOF/开发/校准/最终评价链，不是新的真实 benchmark。CLI 合成结果只有 1 个物理场景，程序不生成其 CI。

科学协议须明确 `confirmation_evaluation`，包括每个 family 的全部 training seeds、forecaster/risk artifact IDs、cost report、development policy ID、calibrator ID、baseline、采样/几何/标签/归一化/切片规则。每个比较族必须完整覆盖 protocol seeds；实际 checkpoint 与 OOF producer seed 必须相符。所有父产物以及已完成 calibration claim/report/completion 要同时提供，不能以旧 teacher 或手工结果表替代。

运行时传入 `--protocol`、`--artifacts`、新的 `--output-dir`，显式选择 `--device cpu|mps` 和 `--threads`。启动前锁定最终比较族和实现，之后无选择/调参。每个 candidate/recording 完成后保存带 hash 的缓存，逐场景写 PID 心跳；中断同参数加 `--resume`。已完成 recording 不重新预测；完成任务只核验旧产物，篡改缓存或 calibration 拒绝继续。结果生成后不能因不理想删除 claim 换模型重做“同一次确认”。

报告分开列每个 seed、固定 seed 平均误差、样本标准差、whole-scene paired bootstrap 和单个 calibration-selected policy。三个 seeds 不等于三倍独立场景，也不是 ensemble prediction。positive harm 先逐 seed 计算再平均，easy guard 逐 seed 保留；配对区间是未做多重检验调整的描述性结果，不是正式风险界。原始 dataset-local 指标仅按录像单列。详情见 [最终评价实现](confirmation_evaluation/implementation_and_limits.md)。

## 联合介入工程验证

```bash
.venv-pytorch/bin/python scripts/check_m3w_joint_intervention.py
.venv-pytorch/bin/python scripts/audit_m3w_causal_recordings.py --report-dir outputs/publication_readiness_2026_09/joint_intervention
.venv-pytorch/bin/python -m pytest tests/test_m3w_joint_intervention.py tests/test_m3w_causal_recordings.py tests/test_m3w_recording_lineage.py tests/test_stage44_worldcore.py tests/test_stage42_source_level_ucy_full_waypoint_integration.py -q
```

第一条检查小规模合成决策与穷举的一致性，以及真实轨迹的共同坐标还原，不读取 future labels 或计算预测准确性。第二条审计包括新的坐标转换元数据，确认它也不受未来位置变化影响。第三条是隔离后的针对性回归测试，本轮 48 passed；不等于全套旧测试通过。

`select_interventions` 接受外部提供的 predicted gain/harm 和 explicit budgets；不会自行选择正式风险预算。求解失败/未最优时返回 baseline。风险筛选接口不验证 IID 或策略是否预先冻结，不能把接口接受误称为获得物理安全证书。详见 [方法与验证边界](joint_intervention/method_and_checks.md)。正式训练仍待主协议和独立校准用途确认。

## 真实训练与恢复探针

```bash
.venv-pytorch/bin/python scripts/probe_m3w_training_runtime.py --run-id cpu4 --threads 4 --steps 240
.venv-pytorch/bin/python scripts/probe_m3w_training_runtime.py --run-id cpu4 --threads 4 --steps 4000 --resume
.venv-pytorch/bin/python scripts/probe_m3w_training_runtime.py --run-id cpu8 --threads 8 --steps 4000
.venv-pytorch/bin/python scripts/probe_m3w_training_runtime.py --run-id mps4 --device mps --threads 4 --steps 240
.venv-pytorch/bin/python scripts/probe_m3w_training_runtime.py --run-id mps4 --device mps --threads 4 --steps 4000 --resume
```

这些 run-id 已有本轮结果。独立重跑时更换 run-id；接着跑时使用原 id、`--resume` 和更大的 `--steps`。不删除或覆盖旧检查点来假装一次新实验。

探针训练一个 68,292 参数的两层 Transformer，1024 个真实历史窗口、batch=128，目标是重建遮蔽的过去位置。**不调用未来标签，不进行预测效果比较，不把 loss 降低算成 M3W 改善。** 它不参与正式模型选择，也不能作为未来独立测试预训练权重。每 60 步原子保存 checkpoint，每 20 步记录 PID、loss、进度；日志、optimizer、模型和随机状态都留在本地缓存。`--checkpoint-every` 可调整保存频率。

结果文件给出有限 loss/梯度、吞吐、内存和恢复后下一次 optimizer step 的参数差异。恢复检查通过不等于 12 小时稳定性保证。MPS 首次在受限沙箱中报了不适用的 macOS 版本错误；系统实际为 macOS 15.3.1，沙箱外真实 MPS 训练成功，不能把前一次错误归因于机器不支持 MPS。脚本不会悄悄切 CPU。

## CREATE 现状

本轮只读 SSH 尝试到达 CREATE 入口，但返回 `Permission denied (publickey)`，入口同时提示门户 MFA。尚未读取 M3W 的远程目录、scheduler、partition 限制和历史产物，**没有提交新作业**。不能把认证错误解释为没有远程任务。具体 M3W 目录仍待用户确认。

按 [CREATE 官方连接文档](https://docs.er.kcl.ac.uk/CREATE/access/) 完成门户 MFA 和已有公钥登记；不要把密码、私钥或登录令牌写入仓库。恢复连接后先只读检查：

```bash
squeue --me
sinfo -o '%P %l %D %G'
sacct -u "$USER" --starttime 2026-09-01 --format=JobID,JobName,State,Elapsed,ExitCode
```

再核对用户指定项目目录的代码版本、环境、config、checkpoint、日志和结果身份。在知道队列资源和现有任务前，不选择 GPU 分区、不重提已有作业。训练必须通过 scheduler，保存 job id、资源、日志、显式时间限制及恢复命令。正式模型训练入口尚待科学协议确定，不能把这份工程探针当作完整 HPC 实验入口。

## 放置实验的原则

当前小模型本地可运行，优先本地完成最小假设检验。正式模型应单独试跑后估计总成本；不能把这个短序列探针的吞吐直接外推到大多智能体模型。数据/环境已经在 CREATE、显存不足或多 seed 成本明显不合理时，再根据核实的资源迁移；当前约 102 GiB 可用磁盘仅是本轮快照，转换前仍需重新检查。

## 测试与研究产物隔离

旧版全套 `python -m pytest tests` 包含真实集成训练、bootstrap 和写报告步骤，不能当作无副作用的 smoke test。部分测试会改写根目录研究状态和历史输出。本轮发现后修复了一个 Stage42 集成测试的临时目录隔离；其余旧测试尚未全部改造，不能声称全套测试已隔离。

新代码优先运行 README 列出的针对性测试。必须跑旧全套时，先保存工作区报告、检查点和研究状态，保留已暂存/未暂存的用户工作；把测试产生的重放产物与正式实验分开。不能将旧测试重新生成的 `pass` 或指标当作独立预测证据，也不能用测试输出覆盖已冻结的论文结果。

Stage5C、SMC 均关闭；没有 metric、seconds-level、true-3D 或 foundation 声明。
