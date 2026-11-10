# M3W 项目交接：给 Claude Code

更新时间：2026-11-10，Europe/London。本文件是本机工作交接入口，不是论文或匿名补充材料。
依据本地文件、Git、真实实验输出和本轮 CREATE 只读查询整理。未重新运行全部历史阶段。

## 最新续接：已完成 policy-matched donor 实验

以下更新发生在最初交接提交 `bd5134a30` 之后。此前第 6 节的
incumbent-slate 结果仍保留，不被新结果替换；现在应先读本节。

- 新代码：`scripts/run_m3w_opendd_onpolicy_donors.py`、
  `src/world_model/m3w_onpolicy_donors.py`、
  `scripts/audit_m3w_onpolicy_donors.py`、
  `scripts/diagnose_m3w_onpolicy_gap.py`。
- 新公开证据：`outputs/publication_readiness_2026_09/opendd_onpolicy_donors_v1/`。
  重点读 `README.md`、`readout.json`、`audit.json`、`generalization_diagnostic.json`。
- 私有权重/标签缓存：`data/stage_cvpr2027_experiments/opendd_onpolicy_donors_v1/`，约 107 MB，未提交。
- `fresh_run`：105 个三重排除候选排序器、63 个双重排除 recording count head、
  42 个选择头，共 121,800 updates。单机原生 arm64 CPU4、workers0 完成。
  没有新增 CREATE 作业；本次续接没有再次查询远端任务。
- `cached_verified`：原有候选、外层分配、101 个 eligible views、780 次 forecast edits
  与旧结果一致。新 21 组模型选择做了零更新恢复，逐项重放一致。
- 训练前登记提交 `25c4c42b7`，全部选择冻结提交 `f9daea2d0`，二者均早于本实验评分。
- 结果：paired optional full-grid 47 → 52，但 incumbent 为 55；
  own-mask uniform 为 49.1784，配对差的 95% CI 跨零。主假设仍失败，不部署。
- 新 paired model 救回 10、破坏 13，切换 82/101。种子结果 17/20/15，
  incumbent 为 19/17/19，不能挑 seed 29 宣称成功。
- 训练内所选动作预测/实际原始通过率为 44.73%/45.18%；held location 为
  67.20%/20.79%。训练任务能够拟合，跨地点评分泛化明显不足。
  这不是已校准部署概率，也不是 full-grid 通过率。
- 29 个相关测试通过，105/63 个模型和 315 个 teacher cache 的嵌套来源核验通过。
  一个 teacher 求解未定、两个重复 donor view 的分配数值校验失败，按协议排除并保留记录。
- `not_run`：新外部独立确认、新风险放行、论文 PDF/作者 Word 合并、Stage5C/SMC。
  不把这个开发集点估计改善写成新的跨数据集成功。

复现时依次运行 runner 的 `register`、`producers`、`fit`、`score`，再运行两个
audit/diagnose 脚本；完整命令在新 README。已完成权重会恢复为零 updates。
所有本轮训练、重放、评分、诊断进程均正常结束，没有任务需要接着等。

下一优先项已经从“补 donor producer”变为：在 donor-held 验证中检查低容量的
keep-versus-switch advantage 模型是否能同时改善排序和概率可靠性，再冻结新外层选择。
不能根据这次 outer 结果直接挑 keep threshold。原 2% 规则与外部确认角色不变。

## 1. 先看这几条

1. 项目根目录：`/Users/yangyue/Downloads/World`。下文相对路径均以此为根。
2. 主 GitHub：<https://github.com/PhDLeYanggggg/World-model>，分支 `main`。
3. 当前论文证据主线：`outputs/publication_readiness_2026_09/`。不要把早期 Stage37/43 的成功标题直接当作当前独立验证结论。
4. 作者正在编辑 Word：`data/stage_cvpr2027_experiments/submission_admin/local_handoff_v218_color_figures/paper_editable.docx`。不要覆盖、自动合并或恢复作者删除的文件。
5. 当前独立论文候选在 `data/stage_cvpr2027_experiments/submission_admin/local_handoff_v219_athens/`。它不等于作者 Word 的最新内容，也不等于已上传。
6. 最新实际工作是上面的 policy-matched donor 实验；第 6 节记录其直接前序 incumbent-slate 实验。两轮主假设均未通过，没有改变部署策略。
7. 仓库有约三千项与当前工作无关的暂存变更，主要是数据湖/WorldCore 及历史数据整理。不要 `git add .`、批量取消暂存、清理未跟踪文件或恢复整个工作树。
8. 本轮训练和评分进程已正常退出，四个已登记 CREATE 作业也为 COMPLETED。不要因旧 heartbeat 文件仍存在就重启任务。

建议阅读顺序：本文件最新续接 → `README_RESULTS.md` 顶部 → 最新实验 README → `reproducibility/M3W_PAPER_REPLAY_ZH.md` → v219 论文与来源核验。
`README.md` 和旧的 `continuation_handoff.md` 包含时间较早的状态，必须按日期和冻结文件核对，不能只读第一段判断现在进行到哪里。

## 2. 目标、当前结论与评价边界

长期目标是可靠的真实世界多模态多智能体世界模型。当前具体研究问题是：强运动基线已经处理大多数简单样本时，怎样学到神经预测的相对增益和伤害，并通过场景级决策选择介入对象、控制相对基线的误差增加？

当前主协议是观察 8 步、预测 12 步；旧 raw-frame t+50/t+100 是补充历史轨道，不能与 8/12 指标直接合并。不同数据集的原生步长不自动代表相同秒数。没有校准证据时不写米、秒、true 3D、foundation 或物理安全。

已经完成大量工程、模型和实验工作，但**尚未证明场景联合选择具有稳定的独立外部优势，也没有完成可部署联合策略的证据闭环**。不能用工程测试通过、事后 oracle、零放行或早期污染结果替代这项证据。

已批准且仍有效的约束：

- 原定 2% 风险规则保持不变。空 easy 参考和空同预算对照应是不可估，不是安全通过。
- 作者明确选择保留原规则、先改模型；没有采用额外的“恒等回退认证”放宽解释。
- 训练、选择、校准、确认角色分开。预处理、teacher、原型和候选生成器遵守同一划分。
- 不用 future endpoint 或 future goal 作输入，不用 central velocity，不用测试终点建 goals。
- 重叠窗口和不同 seed 不能当作新的独立场景；统计以来源组、录像或地点配对。
- Stage5C execution 和 SMC 未授权执行，继续关闭。脚本存在不意味着允许调用。
- 结果分别标记 `fresh_run`、`cached_verified`、`not_run`；使用过来改方法的开发数据不会重新变成独立测试。
- 最终投稿由作者确认。本次不上传、不提交、不修改 OpenReview，也不替换作者的声明文字。

## 3. 代码库与模块位置

### 3.1 主仓库

只有这里是本轮主要编辑、训练和提交的项目仓库：

```text
/Users/yangyue/Downloads/World
https://github.com/PhDLeYanggggg/World-model.git
```

| 路径 | 用途与接手建议 |
| --- | --- |
| `src/world_model/` | 当前研究核心：预测、gain/harm、选择、联合分配、风险、数据适配与统计。626 个非缓存文件，按具体 runner 的 import 链阅读，不要全量扫进上下文。 |
| `scripts/` | 当前实验入口、冻结、训练、评分、诊断、复现与论文构建；2,778 个文件。最近工作都在这里，不是根目录的旧 stage runner。 |
| `configs/` | 424 个配置/作业文件；有些新小实验把固定参数放在 runner + protocol + registration 中，没有独立 YAML。 |
| `tests/` | 2,121 个非缓存文件；包含多代研究和数据湖任务。本轮只跑了明确列出的 24 项相关测试，未声称全库通过。 |
| `reproducibility/` | 135 个文件；中文运行教程、匿名 replay 验证器、材料构建辅助。 |
| `src/m3w/` | 初期 M3W 通用实现：dataset/models/train/eval/gates/evidence/token_schema；其中 `numpy_backend.py` 是旧替代路径，不能当成真实 Torch 训练。 |
| `src/models/`, `src/models/jepa/`, `src/training/` | 早期阶段模型、JEPA、selector 与训练逻辑。 |
| `src/data_unification/`, `src/io/`, `src/features/` | 轨迹转换、SDD memmap 快缓存、episode index、因果特征。 |
| `src/scene/` | scene packs、train-only goals、raster/SDF。标签多为自动 silver，不是人工 gold。 |
| `src/evaluation/` | 早期阶段评估/gates；当前实验也常在 scripts 中直接评分。 |
| `src/inference/` | 历史 final model 和 Stage17 推理。`smc_rollout.py` 存在但不应启用。 |
| `src/stage21_*` 至 `src/stage43_*` | 历史阶段管线，寻找旧结果时使用，不自动代表当前最佳可部署模型。 |
| `src/simulation/`, `src/physics/`, `src/App.jsx`, `src/styles.css` | 本地仿真/界面分支；不把仿真成功当真实数据成功。 |
| `src/worldcore/`, `worldcore/`, `src/data_lake/`, `src/cloud/`, `cloud/` | 另一个 WorldCore/数据湖/云计算工作分支，含大量他人或并行工作暂存变更。本轮没有修改其内容。 |

上面的计数是交接时文件系统快照，不代表每个文件都已审计。可用 `rg --files scripts src/world_model -g '*主题*'` 定位，而不是根据文件名里的 final、verified、success 推断状态。

### 3.2 原始 M3W 架构入口，保留但不是当前主实验

```text
run_train_m3w_world_model.py
run_eval_m3w_world_model.py
run_m3w_gates.py
configs/m3w_small.yaml
src/world_model/m3w_token_schema.py
src/world_model/m3w_batch.py
src/world_model/m3w_dataloader.py
src/m3w/models.py
src/m3w/train.py
src/m3w/eval.py
outputs/m3w/
outputs/m3w_neural_v1/
outputs/m3w_stage28/
outputs/m3w_state_machine/
```

这些实现覆盖 JEPA-only、Transformer-only、Hybrid 和 downstream heads。历史记录里 JEPA 有 non-collapse 但无可靠 downstream lift，神经 residual 也未稳定优于受保护选择器。不要不看对应配置和 schema 就启动整个旧管线。

### 3.3 本轮核实的嵌套第三方 Git 仓库

| 本地位置 | 远程 |
| --- | --- |
| `/Users/yangyue/Downloads/World/external_data/OpenTraj` | <https://github.com/amiryanj/OpenTraj.git> |
| `/Users/yangyue/Downloads/World/external_data/WEPDTOF_source_probe/RAPiD` | <https://github.com/duanzhiihao/RAPiD.git> |

只在 external_data、experiments、worldcore 的三层深度内查了嵌套 `.git`；这不是整台电脑或全部第三方拷贝的穷尽清单。不修改第三方源码来美化基线结果，不把这些数据目录整体提交到主 GitHub。

## 4. 数据、特征、模型权重和结果在哪

### 4.1 当前主实验采用“公开轻量结果 + 本地私有大产物”双目录

```text
outputs/publication_readiness_2026_09/<实验名>/
    README.md / protocol.md / registration.json
    head_freeze.json / readout.json / receipt.json 等轻量记录

data/stage_cvpr2027_experiments/<实验名>/
    数据缓存 / donors / .pt 权重 / 已冻结选择 / heartbeat / process.lock
```

不是每个实验都有以上全部文件，以它的 runner 和 README 为准。`run.checked`、`run.once`、registration 和 freeze 通常绑定具体文件哈希；不要直接修改冻结配置后覆盖旧输出。新科学设计用新版本目录。

单独 clone GitHub **不能**恢复所有原始数据、权重和私有缓存。当前机器保留了这些资产。checkpoint 一般在各实验自己的 data 目录，不是一个通用 `best_model.pt` 可以代表所有结果。模型必须与其输入 schema、split、forecast bank 和冻结策略一起使用。

### 4.2 SDD 历史数据链

```text
external_data/StanfordDroneDataset/
external_data/OpenTraj/
data/stage21_sdd_world_state/
data/stage22_sdd_scene_packs/
data/stage22_sdd_episodes/
data/stage23_sdd_splits/
data/stage24_sdd_fast_cache/
data/stage24_sdd_medium_index/
data/stage26_sdd_feature_store/
outputs/reports/report_stage26_final.md
outputs/reports/stage26_sdd_benchmark_metrics.json
```

这是旧 SDD selector 路线。历史 Stage26、Stage28/30、Stage37 曾报告正结果，但后续发现部分历史 split/teacher/测试选择问题。应按 `paper_evidence_lineage_v1` 和对应重审计决定使用范围，不直接恢复早期“deployable”结论。

### 4.3 外部原始数据，均在项目 external_data 下

```text
external_data/OpenDD_source_audit/
external_data/DUT_author_raw/
external_data/DroneCrowd_annotations/
external_data/DroneCrowd_images/
external_data/DroneCrowd_release_metadata/
external_data/WEPDTOF_source_probe/
external_data/Songdo_Traffic_source_audit/
external_data/pNEUMA_source_intake/
external_data/SwissRoundabouts_source_audit/
external_data/Argoverse2_mf_train_source_audit/
external_data/Argoverse2_mf_train_expansion_v1/
external_data/EuropeanSquares_source_audit/
external_data/TUMDOT_ING_source_audit/
external_data/PDESTRE_source_audit/
external_data/PersonPath22_source_intake/
external_data/VisDrone_MOT_train_source_audit/
external_data/pretrained/
```

目录存在不等于已正式评价。下载、转换、预测准备、合成评分演练、真实评分、风险放行是不同状态。

## 5. 已做的路线，以及哪些结论成立

### 5.1 历史路线概览

| 路线 | 已做工作 | 当前应如何解释 |
| --- | --- | --- |
| SDD scaffold 与数据效率 | world-state shards、scene packs、lazy episodes、fast cache、medium index、因果特征 | 工程基座存在；Stage23 quick-plus 不变成 medium；数据角色和历史指标需沿用后续审计。 |
| selector | 从 hard classification 转 expected-FDE/regret、分 horizon、failure-assisted、保守回退 | 强基线下的核心方向；历史 SDD/UCY 的正结果不能自动证明独立外部泛化。 |
| 坐标/域对齐 | normalization、CORAL/adapter、relative error、外部几何和 goals | 多次缩小分布差异但未证明预测收益，0.0 fallback 不是正迁移。 |
| 历史窗口与原型 | past-only K=8/16/32/64、scene-agnostic prototypes、t50 switchability | 历史 Stage37 有局部正信号，后续证据以当前协议重新界定。 |
| 神经 dynamics | Transformer、JEPA、Hybrid、teacher distillation、bounded residual | 确有真实训练；没有可用证据支持把模块组合本身写成创新或稳定部署优势。 |
| 因果 8/12 + 强公开对照 | CV/OLS、EqMotion、Transformer、Social-STGCNN、分组统计 | 当前论文比较基座；必须统一观察长度、预算和有效标签。 |
| 场景联合选择与同预算对照 | 同样预测器/介入数/位移预算，比较独立、同族、联合和事后上限 | 主问题已变清楚，但学得的联合策略未稳定兑现事后机会。 |
| 真正外部验证 | WEPDTOF、Songdo、Athens 的冻结预测、角色和真实评分 | 已有跨来源证据，但不支持“联合优于独立、可安全部署”的强结论。 |

更早工作总表：`README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md`、`README_M3W_MASTER_SUMMARY_2026_05_27_ZH.md`、`README_RESEARCH_HISTORY_2026_09.md`。这些是历史索引，不能覆盖新的失败或来源审计。

### 5.2 当前主要证据目录

以下都相对 `outputs/publication_readiness_2026_09/`：

| 目录 | 当前事实 |
| --- | --- |
| `paper_evidence_lineage_v1/` | 核对主要 frozen 实验的数据来源与训练角色，不是全历史库无泄露证明。OpenDD 7 地点、202 录像、3,395 个录像内不重叠 query；三种子为 606 重复视图。 |
| `opendd_full_grid_count_motion_v1/` | 同数量/位移预算完整网格事后机会仍明显高于学习分配；oracle 不是真实模型成绩。 |
| `opendd_exchange_attrition_v1/` | 选择错误主要与风险/支持有关；不是没有候选收益。 |
| `opendd_listwise_exchange_v1/` | 63 个新选择头已训练；pointwise/full-grid pass 53，listwise 48/47，均未超过 incumbent 55。 |
| `opendd_incumbent_slate_v1/` | 本轮新完成，详见下一节；允许 keep 仍未修复。 |
| `wepdtof_external_transfer_v1/`, `wepdtof_penalty_breakpoint_v1/` | 冻结外部验证；联合实际零修改；完整固定提案风险与双效用检查 0/45。 |
| `songdo_calibrated_20261110/` | 40 趟校准、160 趟独立确认，3 个校准来源组件和 14 个确认组件。联合固定提案 53/480 通过，但实际联合放行 0；独立规则放行 15、其中 4 通过。 |
| `pneuma_confirmation_v1/`, `pneuma_date_robustness_v1/` | Athens 四天零样本已真实评分，643 query、三种子。对 CV 有小正信号，对独立选择未证明优势；没有 Athens 校准或部署放行。 |
| `external_budget_comparison_v1/` | 三外部域同预算汇总；不能跨 cohort 合并单位或把缺失标签 envelope 当 CI。 |
| `dronecrowd_calibration_readout_v1/` | 15 组校准 + 30 组确认协议保留；校准 gate 失败，所核验注册目录没有确认 readout，不擅自打开保留标签。 |
| `swiss_roundabouts_readout_rehearsal_v1/` | 准备/演练不能当已评分；不要借用 Athens 的正式评分状态。 |

Athens：joint 对 CV 缺失标签包络为 `[+0.02383, +0.08858] pp`；joint 对 independent 为 `[-0.04427, +0.05304] pp`；逐步等位移后为 `[-0.03823, +0.04511] pp`。四天都是一个城市内日期，不是四个独立城市。区间、包络及固定提案/实际放行要分开写。

## 6. 我最后实际做了什么：incumbent-slate 完整交接

### 6.1 核心文件

```text
src/world_model/m3w_incumbent_slate.py
scripts/run_m3w_opendd_incumbent_slate.py
scripts/audit_m3w_incumbent_slate.py
tests/test_m3w_incumbent_slate.py
outputs/publication_readiness_2026_09/opendd_incumbent_slate_v1/
data/stage_cvpr2027_experiments/opendd_incumbent_slate_v1/
```

公开输出包含 `protocol.md`, `registration.json`, `runtime_amendment.json`, `head_freeze.json`, `readout.json`, `failure_diagnostic.json`, `README.md`。
本地权重为 `rdb1_17_point.pt`、`rdb1_17_pair.pt` 等，共 7 地点 × 3 seeds × 2 模型 = 42 个；另有 21 个 `*_choices.json`。

### 6.2 设计与执行

假设：此前强制交换可能损害原方案，因此把“保持现有分配”也作为候选，并比较 pointwise 与增加 pairwise loss 的模型。
两个 48/32 MLP，400 AdamW updates，batch 32 slates；总计 16,800 updates。不是再训练一个端到端大模型。
同一模型同时产生 forced 与 optional 选择。只在原 101 个 eligible 视图动作，保持 780 次轨迹修改、原位移上限及 2% 规则。
选择全部冻结之后才评分，无 held-site threshold 搜索。

本轮修复并完成：

1. 空 donor 列表导致 `torch.stack([])`；修成返回空布尔目标。
2. 恢复登记错误地把已解析对象当 JSON 文本；修成比较归一化对象。
3. 保留原登记和四个已完成 cells，以 `runtime_amendment.json` 绑定代码变更；两处修复均早于评分。
4. 继续剩余 13,600 updates，57.19 秒完成。最初 pilot/中断运行已完成 3,200 updates。
5. 再跑完整 fit：21 cells、0 updates、13.14 秒，所有选择与已有文件一致。
6. 冻结 commit `b7c6a6c39` 先于评分；评分 9.52 秒，0 updates。
7. 用 checkpoint 重建分数和选择，逐项核对，再做标签/校准诊断。
8. 24 个相关测试通过，不是全部 2,121 个测试文件全部通过。

### 6.3 结果与具体失败原因

| 策略 | allocation exchange 数 | full-grid 双通过 / 606 |
| --- | ---: | ---: |
| 不改变现有 allocation | 0 | 55 |
| 原 aligned ranker | 101 | 44 |
| pointwise forced | 101 | 46 |
| pointwise optional | 95 | 47 |
| paired forced | 101 | 45 |
| paired optional | 94 | 47 |
| paired optional 相同 switch mask 下均匀选择 | 94 | 49.0008（期望） |
| 同一 eligible mask 上含 keep 的 hindsight oracle | diagnostic | 100 |

paired optional 对 incumbent 的地点等权 full-grid 通过率差为 `-1.215 pp`，95% CI `[-3.120, 0.000]`；对相同 switch mask 的 uniform 为 `-0.549 pp [-1.481, -0.026]`。使用 3,000 次地点配对 bootstrap，先在地点内平均 seeds。主假设失败，未部署新模型。

主要诊断：

- 仍交换 94/101；救回 5 个 full-grid pass，破坏 13 个。
- 七个净损失来自 rdb1，一个来自 rdb3；其余地点没有净提升。
- eligible 视图中，keep 的预测 sigmoid 均值 5.36%，真实原始通过率 26.73%；所选动作预测 38.00%，实际 18.81%。这只是未校准分数诊断。
- 训练 donor 中，keep 正标签 473/4,507=10.49%；其中 2,678 slate 全负、19 全正，仅 1,810 heterogeneous 能提供成对标签差异。
- 训练 donor incumbent 来自固定优先/随机 schedule，评估 incumbent 却是学习分配。这是可核实的分布差异，尚不能声称它是唯一因果原因。
- pairwise 与 pointwise optional 的 40 个动作不同，但地点汇总 full-grid 一样；换 loss 没有兑现优势。
- 特征已有 allocation-weighted context；不能误说“完全没有 allocation 信息”。

这些数据是已暴露开发证据。所有外部冻结策略、论文 Word、声明、Stage5C 和 SMC 均未改变。
沿用特征会在同一录像内汇总多个 query 的已观测历史，因此本实验是回顾性批量分配，不能据此声称逐时刻在线因果部署；它不读取 held future target 作为特征。

### 6.4 直接可运行的复现命令

```bash
cd /Users/yangyue/Downloads/World
export LC_ALL=C
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_incumbent_slate.py fit
.venv-pytorch/bin/python scripts/run_m3w_opendd_incumbent_slate.py score
.venv-pytorch/bin/python scripts/audit_m3w_incumbent_slate.py
.venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_incumbent_slate.py \
  tests/test_m3w_exchange_listwise.py \
  tests/test_m3w_exchange_risk_heads.py \
  tests/test_m3w_exchange_attrition.py \
  tests/test_m3w_opendd_full_grid_exchange.py \
  tests/test_m3w_exchange_context.py \
  tests/test_m3w_one_exchange_ranker.py
```

权重完整时 `fit` 应为 0 updates；它不是必须重新训练。`score` 重用 frozen choices，不选择新阈值。
该冻结 scorer 的 Counter 聚合会省略零值键，读取时用 `.get(key, 0)`。它对当前非空、material-positive 对照已核验，不是空基准的通用 scorer；新版本如扩展到全零对照，要先修复并测试零值键处理，不偷偷改旧冻结结果。

## 7. 论文、Word、补充材料与匿名复现

| 内容 | 绝对路径 |
| --- | --- |
| 作者正在编辑的 Word | `/Users/yangyue/Downloads/World/data/stage_cvpr2027_experiments/submission_admin/local_handoff_v218_color_figures/paper_editable.docx` |
| 最新独立候选正文 PDF | `/Users/yangyue/Downloads/World/data/stage_cvpr2027_experiments/submission_admin/local_handoff_v219_athens/paper.pdf` |
| 最新独立候选补充 PDF | `/Users/yangyue/Downloads/World/data/stage_cvpr2027_experiments/submission_admin/local_handoff_v219_athens/supplement.pdf` |
| 匿名汇总复算包 | `/Users/yangyue/Downloads/World/data/stage_cvpr2027_experiments/submission_admin/local_handoff_v219_athens/replay.zip` |
| 同包 manifest | `/Users/yangyue/Downloads/World/data/stage_cvpr2027_experiments/submission_admin/local_handoff_v219_athens/manifest.json` |
| LaTeX 源 | `/Users/yangyue/Downloads/World/data/stage_cvpr2027_experiments/evidence_manuscript_v219_athens_confirmation/manuscript.tex` |
| 源码压缩包 | `/Users/yangyue/Downloads/World/data/stage_cvpr2027_experiments/evidence_manuscript_v219_athens_confirmation/main_source.zip` |
| 构建/渲染/保护检查记录 | `/Users/yangyue/Downloads/World/outputs/publication_readiness_2026_09/evidence_manuscript_v219_athens_confirmation/` |
| 构建入口 | `/Users/yangyue/Downloads/World/scripts/build_m3w_athens_candidate.py` |
| Athens 汇总验证器 | `/Users/yangyue/Downloads/World/reproducibility/m3w_athens_replay/verify.py` |
| 中文教程 | `/Users/yangyue/Downloads/World/reproducibility/M3W_PAPER_REPLAY_ZH.md` |

v219 主文 8 页、补充 11 页，先前已渲染检查。正文和表格黑白，代码生成图保持原始配色。主文 CVPR review PDF 有模板行号；Word 编辑版没有行号。不要为了“去 AI 味”捏造图或改实验数值。

v219 已加入 Athens，但**本轮 incumbent-slate 负结果尚未整合进 PDF 或 replay.zip**，只是进入代码、结果记录和本交接文件。作者 Word 也没有自动合并 v219 新段落。

交接时再次核对 SHA-256：

```text
Word:       3607c46daaae877dcc58c04d26d1fa8f629f9c3365bd82673a22118c665b5f95
paper.pdf:  f62dd93a313e8c0f511b9fc3f2f7782ce4a14115121b3427375e1225949c6e75
supplement: cf79cd378ec123eddc23f1f5354bfc887b44ae91078ea1a8d9bd905029660931
replay.zip: 026fee32f6bf33e3b1fc26a5732236cfb02a108bf31de2bf4beb08b51adccf2a
```

旧 v218 loose PDF 被作者移除；原补充内容曾从它的已核验 replay.zip 读取，没有恢复删除文件。
不要修改用户的 Word 来覆盖这些变化。论文中的 AI 工具使用声明应依据实际使用记录与作者核对，本轮未修改。

匿名包解压到新空目录后，运行 `python3 -I -B -S verify.py`。这只重算已冻结汇总，不读取原始轨迹、不重训，也不是独立研究者复现。

## 8. 本机环境与 CREATE HPC

### 8.1 本机

- Apple M3 Max，48GB；主解释器：`/Users/yangyue/Downloads/World/.venv-pytorch/bin/python`，原生 arm64。
- 历史卡死根因是 x86_64 Conda/Intel MKL/OpenMP 与 Apple Silicon 训练初始化 SHM 路径不兼容。不要改回默认 Conda。
- 当前真实训练已跑通 CPU 4 threads、interop 1、DataLoader workers 0；不要只用 `import torch` 判断训练成功。
- 当前脚本有 Darwin/arm64 入口检查，checkpoint 每 100 updates 原子保存 model/optimizer/RNG。`process.lock` 防重复写入。
- user 允许慢跑，只要没有卡死且合理控制在 12 小时内；不因慢自动伪装为 quick/medium 完成。
- 本机部分 Perl/Git 工具在 `C.UTF-8` 下报 locale 错，命令使用 `LC_ALL=C`。本轮 `shasum` 在此设置下正常。

文档工具不一定在训练 venv：可用的 Python 文档环境曾位于 `/Users/yangyue/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`；需要时先检查存在性。Tectonic 位于 `/Applications/ChatGPT.app/Contents/Resources/tectonic/tectonic`。不要为文档排版改坏训练环境。

### 8.2 CREATE

SSH 连接参数保存在本地私有文件：

```text
data/stage_cvpr2027_experiments/create_handoff_20260923/observations.json
data/stage_cvpr2027_experiments/create_handoff_20260923/compute_handoff_20260927.json
```

本文件不复制密码、私钥内容或令牌。连接主机是 `hpc.create.kcl.ac.uk`；主项目逻辑根是 `/users/k24101830/m3w/`。Ceph 实际挂载可能不同，使用已有路径核验，不随意替换根目录。

2026-11-10 只读确认存在的远端代码目录：

```text
/users/k24101830/m3w/european_temporal_auxiliary_v1/code
/users/k24101830/m3w/european_easy_risk_priority_v1/code
/users/k24101830/m3w/european_easy_harm_deviance_v1/code
/users/k24101830/m3w/european_easy_component_diagnostic_v1/code
/users/k24101830/m3w/european_easy_hurdle_v1/code
/users/k24101830/m3w/european_selected_pool_v1/code
/users/k24101830/m3w/european_easy_gradient_diagnostic_v1/code
/users/k24101830/m3w/european_selected_set_calibration_v1/code
/users/k24101830/m3w/european_task_risk_v1/code
/users/k24101830/m3w/european_boundary_diagnostic_v1/code
```

已登记 runtime：`/users/k24101830/m3w/easy_hurdle_runtime_v2/venv/bin/python`，本轮没有重建或重新验证该 runtime。受保护的另一个项目根是 `/users/k24101830/fully_synthetic_stepC_c4_v2`，不能把 M3W 依赖装进去或改它的作业。

本轮 `sacct` 实际返回：

| Job | State | Exit | Elapsed |
| --- | --- | --- | --- |
| 37835856_0 | COMPLETED | 0:0 | 00:04:09 |
| 37835859 | COMPLETED | 0:0 | 00:00:18 |
| 37837636 | COMPLETED | 0:0 | 00:03:40 |
| 38287008 | COMPLETED | 0:0 | 00:00:06 |

这里只核对了这些已知作业，不代表账户没有其他项目任务，也不代表现在 GPU 空闲。没有提交、取消或重启任务。
远端使用说明见 `outputs/publication_readiness_2026_09/european_temporal_auxiliary_v1/create_reproduction_zh.md` 和 `create_accounting_recovery_20261108/README.md`。其中旧“排队/未跑”段落是历史记录，不能覆盖新的 sacct 终态。

管理入口：`scripts/manage_m3w_temporal_auxiliary_create.py`、`scripts/read_m3w_temporal_auxiliary_create.py`。运行新任务之前先检查既有 submission receipt、squeue、sacct；SSH 失联不等于任务终止，不能因此重复提交。训练只能用调度器，不在登录节点跑。

## 9. Git 状态与安全接手

交接前的关键提交：

```text
cbefd91b4  v219 Athens 证据进入独立主文/补充/replay
aceb591da  incumbent-slate 训练前协议
b7c6a6c39  42 模型和选择冻结，先于评分
```

本交接文件、最新失败 readout/diagnostic、24 项测试及 README/state 更新会作为单独限定范围提交。实际最新 commit 以 `git log -3 --oneline` 为准。

工作树不是干净状态。本轮检查到的数据湖相关暂存包括大量 A/D/AM，约三千项；这是已有用户/并行工作，不属于本轮实验。`research_state.json` 还有大量本轮无关差异。不要为了得到 clean tree 撤销它们。

我采用临时 `GIT_INDEX_FILE` 从 HEAD 建提交，只加入自己的明确文件；`README_RESULTS.md` 和 `research_state.json` 只加入顶部本轮 hunk。提交后仅校准自己且原本未被用户暂存的路径，保留其他索引。Claude 如不熟悉这种方式，先只读检查 diff，不要直接 commit 默认暂存区。

不提交：原始数据、images/videos、fast/feature/history/latent caches、大 episode、checkpoint、第三方仓库、`.venv-pytorch`、凭据。
只提交本轮代码、配置、协议、结果汇总、文档与轻量哈希。不要把此交接文件放入匿名投稿包，它有本地路径和公开仓库身份。

## 10. 现在最值得继续的工作

### 优先级 1：修正 donor incumbent 与评估 policy 的分布差异（已完成本轮检验）

此项现已实现、训练和评分，见本文件顶部最新续接。下面保留原问题背景。

先读 `scripts/run_m3w_opendd_one_exchange_ranker.py` 的 `donor()`：目前训练 incumbent 由固定优先及随机 schedule 生成。再读 `scripts/run_m3w_opendd_exchange_context.py` 的 `evaluate()`：当前评估 anchor 来自冻结学习分配。
最小实验应只改变 donor anchor 生成来源：用与评估一致的策略，在排除 donor 与 outer location 的 producer 上生成，然后比较旧生成方式。不要只调 keep threshold。

关键依赖是 actor/teacher 的嵌套排除是否齐备。已有 rank heads 排除了两侧，不代表新的 allocation actor 也已经双重排除；缺失时补正确训练，不能复用见过 donor 或 held-site 的 actor 来制造 on-policy 改善。
已补正确嵌套训练。结果有点估计改善，但未超过 incumbent；不要重复执行为“未运行”。

### 优先级 2：分清选择效用与风险放行

同预算下优于 incumbent/uniform/independent，才有理由进一步研究校准 release。保留 full-grid risk、简单样本、最差地点、未定义支持和 paired uncertainty。不存在新的外部成功，不打开保留确认集寻找有利均值。

### 优先级 3：完成作者 Word 与证据候选的最终合并

先读取作者修改，不覆盖。v219 已补真实 Athens 结果；本轮新负结果可以在补充中加入紧凑消融，但不是必须为每个小实验增加主文篇幅。核心贡献应围绕可验证的研究问题、协议和实际结果，而不是模块堆叠。

截止时间本轮核对官方网页：登记 2026-11-10 AoE、全文 2026-11-16 AoE、补充 2026-11-23 AoE。
来源：<https://cvpr.thecvf.com/Conferences/2027/Dates>。
格式与声明要求：<https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines>。
本轮没有检查 OpenReview 实时状态；不要把用户之前说已登记当作这轮门户核验。

## 11. 接手后的最短工作清单

1. 确认 cwd、branch、dirty/staged 状态；先看本文件和最新 README_RESULTS，不先跑全部阶段。
2. 读取 `opendd_incumbent_slate_v1/{README.md,readout.json,failure_diagnostic.json}`，理解当前负结果。
3. 确认本地 42 checkpoint 和 21 choices 存在，若需要验证只做原命令零更新 replay，不删除重训。
4. 读取新 policy-matched donor 的冻结与诊断，重点区分训练内拟合和 held-location 泛化；不要重复补已完成的 producer。
5. 如果继续科研，新设计先写独立版本协议再运行，保留失败结果，不碰外部冻结阈值。
6. 如果优先投稿，保护作者 Word，合并已冻结证据后重新渲染检查，最终由作者提交。

交接时没有留下本轮需要等待的训练或评分任务。记录中的 PID 84281、84483、84642 已退出；本轮相关进程检查没有发现仍运行的 incumbent 训练/诊断进程。其他数据湖任务没有被取消或全面审计。
研究总目标未完成；当前不是“投稿准备已完成”。这份交接的作用是让下一位协作者能继续实际工作，而不是从旧成功标题或旧会话记忆重新猜测项目状态。
