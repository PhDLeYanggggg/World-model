# M3W 本地与 CREATE 操作记录

日期：2026-09-16。用途：当前可复现的工程步骤，不是正式预测实验教程的完成版。

## 当前 v6：输入数值范围修复后的完整匹配训练

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
