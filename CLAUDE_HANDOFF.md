# M3W 项目交接：给 Claude Code

更新时间：2026-11-10，Europe/London。本文件是本机工作交接入口，不是论文或匿名补充材料。
用户最新指令：Codex 不再继续研究任务，只把所有任务和已产出内容同步给 Claude。
本次仅整理已有代码、结果和任务状态；没有新训练、新评分、新测试或远端查询。研究目标保持暂停。
根目录 `CLAUDE.md` 指向本文件。这里的“已完成”来自注明的产物，不表示 Claude 已阅读或确认接收。

## 0. 接手快照：先分清已完成与尚未开始

交接提交 `dd3dbfc38` 之后，用户继续了科研目标。四组实验最初登记在
`opendd_nested_advantage_v1`，但该路径出现另一份并行五组草案，训练前哈希检查中止，0 updates。
保留草案不改，四组实验现单独登记为 `opendd_advantage_factorial_v1`。
入口 `scripts/run_m3w_opendd_advantage_factorial.py`，数据角色构建
`scripts/m3w_advantage_factorial_data.py`，模型 `src/world_model/m3w_advantage_factorial.py`。
105 个四重排除 ranker、105 个三重排除 count head、252 个内层选择器和 84 个外层选择器
均已训练完毕，共 260,400 updates；21 组外层选择已冻结并完成评分、来源核查和泛化诊断。
作者已同意改用全部验证录像上的净收益进行内层选择，见新目录
`selection_amendment.md/json`；无交换不计安全通过，原 2% 门槛不变。
最新 27 项针对性测试通过，未跑全库测试；这些是暂停前的结果，本次交接没有重跑。

**最终结论：主假设失败，不部署。** 内层选出的策略 full-grid 为 48，保持原分配为 55，
上一轮 paired selector 为 52；新策略救回 7、破坏 14，切换 90/101。
相对原分配的等地点平均差为 -0.26247 pp，3,000 次地点 bootstrap 的 95% CI 为 [-0.78740, 0]。
这些是已暴露 OpenDD 开发数据上的通过计数，不是 FDE 百分比改善，不是独立外部确认。

最新入口与产物，均相对 `/Users/yangyue/Downloads/World/`：

| 内容 | 路径/状态 |
| --- | --- |
| 完整结果及复现说明 | `outputs/publication_readiness_2026_09/opendd_advantage_factorial_v1/README.md` |
| 原始轻量结果 | 同目录 `readout.json`、`audit.json`、`generalization_diagnostic.json` |
| 冻结与选择 | 同目录 `registration.json`、`selection_amendment.md/json`、`inner_freeze.json`、`selection.json`、`head_freeze.json` |
| 训练入口 | `scripts/run_m3w_opendd_advantage_factorial.py` |
| 因果数据/模型 | `scripts/m3w_advantage_factorial_data.py`、`src/world_model/m3w_advantage_factorial.py` |
| 核验/诊断 | `scripts/audit_m3w_advantage_factorial.py`、`scripts/diagnose_m3w_advantage_factorial.py` |
| 本地权重和缓存 | `data/stage_cvpr2027_experiments/opendd_advantage_factorial_v1/`，不在 GitHub |
| 交接完整性清单 | `outputs/publication_readiness_2026_09/claude_handoff_20261110/manifest.json` |
| 待继续工作 | 第 12、14 节；仅记录，不由 Codex 自动执行 |

暂停前所有本轮训练/评分/诊断调用均已正常结束，没有本轮任务需要继续等待。
当前交接不改变任何运行中的第三方或 Claude 任务，也不查询或取消 HPC 作业。

以下 10:53 快照是较早交接历史，不覆盖上面的最终状态。

本次范围是 World/M3W 项目及它引用的本地、远端资产，不是整台电脑所有项目，也不是把原始数据和权重打包进一个文件。

| 项目 | 2026-11-10 10:53 GMT 的核实状态 |
| --- | --- |
| 工作目录 | `/Users/yangyue/Downloads/World` |
| 交接刷新前的代码版本 | `30a6e55c0d56cf344441971f8d77181f63bd97dc`；本地 HEAD 与远端 `origin/main` 一致 |
| 最新完成实验 | `opendd_onpolicy_donors_v1`，训练、冻结、评分、诊断均已完成；主假设失败 |
| 权重与缓存完整性 | 本次重新计算 SHA-256，1,113 个该实验本地产物与记录一致；这是完整性核查，不是重新训练或重新证明无泄露 |
| 模型数量 | 42 个选择器 checkpoint、105 个排序器、63 个 recording count head；21 份冻结选择 |
| 数据缓存 | 315 个 teacher NPZ、126 个 donor NPZ；不提交 GitHub |
| 本地进程 | 本次未发现命令名匹配 `run_m3w/train_m3w/audit_m3w/diagnose_m3w` 的 Python 进程；不代表其他项目或远端没有任务 |
| 运行环境 | `.venv-pytorch/bin/python`，arm64，Python 3.11.1，安装的 Torch 2.12.0；此次只查环境元信息，真实训练证据见既有 runtime 日志 |
| 磁盘 | 可用约 28 GiB；现有 runner 保留至少 10 GiB，不能再无预算扩充缓存 |
| 用户工作保护 | 3,019 项既有暂存变更，另有 README/state 未暂存差异；不要全量提交、清理或回滚 |
| 作者文件 | 第 7 节 Word、paper、supplement、replay 的四个 SHA-256 本次再次匹配；均未修改 |
| 下一实验 | 低容量/相对收益 + 嵌套地点验证仅为待实施方案，`not_run`；没有对应新训练结果 |

本次交接工作只更新本文件、README 交接入口和 research state，不启动下一轮实验、不改变协议、不修改论文或投稿门户。
本文件后续出现的训练次数、29 项测试和 CREATE 终态来自注明来源的已完成记录，不能描述为本次交接重新执行。

**给 Claude Code 的第一条任务：**先读本文件第 0、12、14 节和最新实验 README，确认本机资产与远端历史记录的区别。不要从 Stage37 的历史成功标题直接开始部署，也不要自动重跑所有 Stage。后续执行以用户给 Claude 的任务为准。

## 前序实验：已完成 policy-matched donor 实验

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
6. 最新实际工作是 `opendd_advantage_factorial_v1`；on-policy donor 和 incumbent-slate 为前两轮。三轮主假设均未通过，没有改变部署策略。
7. 仓库有约三千项与当前工作无关的暂存变更，主要是数据湖/WorldCore 及历史数据整理。不要 `git add .`、批量取消暂存、清理未跟踪文件或恢复整个工作树。
8. 最新本轮训练和评分调用已正常退出；四个已登记 CREATE 作业的较早查询为 COMPLETED，本次未重新查询。不要因旧 heartbeat 文件仍存在就重启任务。

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
| `src/world_model/` | 当前研究核心：预测、gain/harm、选择、联合分配、风险、数据适配与统计。627 个可检索非 pyc 文件，按具体 runner 的 import 链阅读，不要全量扫进上下文。 |
| `scripts/` | 当前实验入口、冻结、训练、评分、诊断、复现与论文构建；2,781 个可检索非 pyc 文件。最近工作都在这里，不是根目录的旧 stage runner。 |
| `configs/` | 424 个配置/作业文件；有些新小实验把固定参数放在 runner + protocol + registration 中，没有独立 YAML。 |
| `tests/` | 2,122 个可检索非 pyc 文件；包含多代研究和数据湖任务。最新已完成实验记录为 29 项相关测试通过，前序为 24 项；此次交接未重跑测试，未声称全库通过。 |
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

此次在 external_data、experiments、worldcore 的四层深度内核对了嵌套 `.git`；这不是整台电脑或全部第三方拷贝的穷尽清单。不修改第三方源码来美化基线结果，不把这些数据目录整体提交到主 GitHub。

### 3.4 前序调用链与当前复用依赖

以下文件均位于 `/Users/yangyue/Downloads/World/` 下。

| 顺序与文件 | 关键职责 |
| --- | --- |
| `scripts/run_m3w_opendd_onpolicy_donors.py` | 前序入口，当前 factorial 复用其已冻结依赖；最新入口见第 12 节。`register` 绑定代码/输入，`producers` 生成嵌套训练模型，`fit` 训练并冻结选择，`score` 才读取评价标签。 |
| `src/world_model/m3w_onpolicy_donors.py` | `fit_roster` / `verify_role` 核对排除地点；`score_anchor` 只以分数、合法动作、数量和位移预算生成 incumbent，不接受未来标签。 |
| `scripts/run_m3w_recording_count.py` | `fit` 保存网络、优化器、RNG、loss trace；`load` 检查 checkpoint；`menus` 产生候选；`labels` 为训练数据构建 teacher。 |
| `scripts/run_m3w_matched_bridge_swaps.py` | 上游候选排序训练、因果候选库与配对特征。 |
| `scripts/run_m3w_opendd_exchange_context.py` | `register` 校验上游来源；场景/交互 context 与特征选择。 |
| `scripts/run_m3w_opendd_one_exchange_ranker.py` | recording/query 索引、合法交换特征与 `measure` 评分；评分函数不能搬进推理特征路径。 |
| `scripts/run_m3w_opendd_listwise_exchange.py` | slate 的 padding、特征和冻结 eligible mask。 |
| `src/world_model/m3w_incumbent_slate.py` | `ListRanker`、显式 keep 特征、pointwise/pairwise loss、forced/optional 决策。 |
| `scripts/run_m3w_opendd_incumbent_slate.py` | 前序训练及当前沿用的选择/评价辅助；不是新的独立实验结果。 |
| `scripts/audit_m3w_onpolicy_donors.py` | checkpoint 和嵌套数据角色检查、选择重放、失败切片。 |
| `scripts/diagnose_m3w_onpolicy_gap.py` | 比较训练内与 held-location 的分数、校准误差和实际通过情况。 |

数据流：观测历史与合法候选 → nominee ranker → recording count head → 固定预算 incumbent → 合法 keep/exchange slate → 选择器 → 冻结 choices → 独立评分函数读取标签 → 分地点 bootstrap 与风险检查。

注意这些研究脚本大量沿用 `prior/parent/run/study` 别名，且部分路径是模块级 `PRIVATE/OUT` 常量。不要只看变量名猜来源，不要 monkey-patch 旧模块全局路径来覆盖冻结实验；扩展新版本时显式传入路径与角色。

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

前序 on-policy donor 实验的完整根目录；最新 factorial 目录见第 12 节：

```text
/Users/yangyue/Downloads/World/data/stage_cvpr2027_experiments/opendd_onpolicy_donors_v1/
    rdb1_17_point.pt              # 地点 rdb1 留出、seed17、pointwise 选择头
    rdb1_17_pair.pt               # 同一角色的 paired 选择头；不是可通用部署权重
    rdb1_17_choices.json          # 已冻结的选择；其余地点/seed 同样命名
    producers/rank_*.pt           # 105 个三重地点排除的候选排序器
    producers/count_*.pt          # 63 个双重地点排除的数量预测头
    teachers/*.npz               # 315 个 teacher 标签缓存及相邻 JSON 来源记录
    donors/*.npz                 # 126 个训练 slate 缓存及相邻 JSON 来源记录
    events_*.jsonl               # 每次调用的 PID、更新次数、内存与终态
    heartbeat.json               # 最后心跳；不是进程仍然存活的证明
    process.lock                 # 操作系统锁；文件存在本身不代表被占用

/Users/yangyue/Downloads/World/outputs/publication_readiness_2026_09/opendd_onpolicy_donors_v1/
    protocol.md                  # 冻结目标、比较对象和原 2% 规则
    registration.json            # 输入/代码哈希与训练参数
    producer_freeze.json         # producer 清单及来源
    head_freeze.json             # 21 个地点/seed 单元、权重和 choices 哈希
    readout.json                 # 冻结评分；不是用来再选模型的验证集
    audit.json                   # 嵌套来源、重放、损失有限性等核查
    generalization_diagnostic.json
    runtime.json
    README.md
```

加载上述前序选择头应沿用 runner 中的 `study.load` 和 `ListRanker`，检查 identity/hash，再通过与训练一致的 `with_incumbent` 特征构造调用；最新 factorial 使用 `Selector(arm, norm)`，不可混用。checkpoint 内的 `trace` 是每 100 updates 记录的真实 loss，`step` 是更新次数；`net/opt/rng` 分别保存网络、优化器与随机状态。

`outputs/final_model/`、`outputs/final_model_v1_1/` 和 `outputs/final_model_v1_2/README_FINAL_MODEL_V1_2_DIAGNOSTIC_ONLY.md` 是历史产物，不是最新 OpenDD 的发布包。`src/inference/final_world_model_inference.py` 与 `src/inference/stage17_selector_inference.py` 也属于旧接口；不能将“可调用”混同于已通过当前泛化与风险门槛。

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
| `opendd_onpolicy_donors_v1/` | 训练分配改为 policy-matched 后 paired full-grid 为 52，仍低于原分配 55。 |
| `opendd_advantage_factorial_v1/` | 最新四组实验已训练、评分和诊断；内层选出策略为 48，四个单臂 51/48/50/51，主假设失败。 |
| `wepdtof_external_transfer_v1/`, `wepdtof_penalty_breakpoint_v1/` | 冻结外部验证；联合实际零修改；完整固定提案风险与双效用检查 0/45。 |
| `songdo_calibrated_20261110/` | 40 趟校准、160 趟独立确认，3 个校准来源组件和 14 个确认组件。联合固定提案 53/480 通过，但实际联合放行 0；独立规则放行 15、其中 4 通过。 |
| `pneuma_confirmation_v1/`, `pneuma_date_robustness_v1/` | Athens 四天零样本已真实评分，643 query、三种子。对 CV 有小正信号，对独立选择未证明优势；没有 Athens 校准或部署放行。 |
| `external_budget_comparison_v1/` | 三外部域同预算汇总；不能跨 cohort 合并单位或把缺失标签 envelope 当 CI。 |
| `dronecrowd_calibration_readout_v1/` | 15 组校准 + 30 组确认协议保留；校准 gate 失败，所核验注册目录没有确认 readout，不擅自打开保留标签。 |
| `swiss_roundabouts_readout_rehearsal_v1/` | 准备/演练不能当已评分；不要借用 Athens 的正式评分状态。 |

Athens：joint 对 CV 缺失标签包络为 `[+0.02383, +0.08858] pp`；joint 对 independent 为 `[-0.04427, +0.05304] pp`；逐步等位移后为 `[-0.03823, +0.04511] pp`。四天都是一个城市内日期，不是四个独立城市。区间、包络及固定提案/实际放行要分开写。

## 6. 直接前序工作：incumbent-slate 完整交接

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

### 6.5 前序 policy-matched donor 的完整命令

以下用于复现已完成的前序实验，不是接手时必须全部再跑。建议先读取公开 README 和 freeze；只有需要核实结果时才执行。已有 checkpoint 完整时训练阶段应恢复为零 updates；若缺失，则会实际补训，不能称纯只读检查。

```bash
cd /Users/yangyue/Downloads/World
export LC_ALL=C OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_onpolicy_donors.py register
.venv-pytorch/bin/python scripts/run_m3w_opendd_onpolicy_donors.py producers
.venv-pytorch/bin/python scripts/run_m3w_opendd_onpolicy_donors.py fit
.venv-pytorch/bin/python scripts/run_m3w_opendd_onpolicy_donors.py score
.venv-pytorch/bin/python scripts/audit_m3w_onpolicy_donors.py
.venv-pytorch/bin/python scripts/diagnose_m3w_onpolicy_gap.py
.venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_onpolicy_donors.py \
  tests/test_m3w_incumbent_slate.py \
  tests/test_m3w_exchange_listwise.py \
  tests/test_m3w_exchange_risk_heads.py \
  tests/test_m3w_exchange_attrition.py \
  tests/test_m3w_opendd_full_grid_exchange.py \
  tests/test_m3w_exchange_context.py \
  tests/test_m3w_one_exchange_ranker.py
```

此 runner 不使用通用 `--resume`：重新运行同一阶段即按 identity 恢复。`--limit-producers N` 仅用于 producer pilot，不能把有限 pilot 当完整矩阵。出现注册或哈希不匹配时先诊断，不能删 registration、改哈希或跳过检查继续。

### 6.6 避免混读数字

- `606 views` 是 202 录像 × 3 seeds，不是 606 个独立场景；不把 seeds 作为独立统计单位。
- `780 forecast edits` 是轨迹修改预算；`82/101 exchanges` 是重新分配动作，两者不是同一个计数。
- `52 full-grid passes` 是满足评价条件的视图数，不是 FDE 改善 52%，也不是外部部署通过率。
- 前序模型训练使用的 `original-dual` 二元标签与最终 `full-grid` 检查不同。最新 factorial 已改用 full-grid 标签并比较相对收益目标，仍未成功，因此不能再把目标错配视为已解决全部失败的解释。
- 最新 paired optional 的原始风险支持仍有 399 个 undefined view；不能将其记为零风险或全部安全。
- oracle 是读取评价标签后求出的机会参考，不能当模型推理结果。训练内 ECE 很低也不能代替 held-location 校准。

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

v219 已加入 Athens，但 **incumbent-slate、policy-matched donor 和最新 factorial 三轮负结果尚未整合进 PDF 或 replay.zip**，只进入代码、结果记录和本交接文件。作者 Word 也没有自动合并 v219 新段落。本次不修改作者文稿或 AI 使用声明。

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

此前 2026-11-10 的 `sacct` 查询记录返回如下；此次交接没有再次查询远端：

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
bd5134a30  incumbent-slate 负结果、诊断与初版交接
25c4c42b7  policy-matched donor 训练前登记
f9daea2d0  新 producer / selector / choices 冻结，先于评分
30a6e55c0  最新负结果、泛化诊断、README/state/交接更新
dd3dbfc38  较早 Claude 交接与本地资产核验
54bfae0cd  factorial 四组实验隔离注册，早于训练
0e78f2de7  producer 完成，保存作者批准的验证分母修订
37b871236  336 个选择器与全部外层选择冻结，早于评分
```

以上均为已完成提交；本次交接开始时 `37b871236` 与远端 main 一致。本次补交暂停前已生成的轻量结果、测试补充和交接文档，不运行实验；最终同步提交以 `git log -3 --oneline` 为准。GitHub 不含本机 checkpoint。

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

此前工作记录核对的官方日期为：登记 2026-11-10 AoE、全文 2026-11-16 AoE、补充 2026-11-23 AoE。此次交接没有重新查询官网；接手后安排投稿前应再查当前页面。
来源：<https://cvpr.thecvf.com/Conferences/2027/Dates>。
格式与声明要求：<https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines>。
本轮没有检查 OpenReview 实时状态；不要把用户之前说已登记当作这轮门户核验。

## 11. 接手后的阅读清单

1. 确认 cwd、branch、dirty/staged 状态；先看本文件和最新 README_RESULTS，不先跑全部阶段。
2. 先读 `opendd_advantage_factorial_v1/{README.md,readout.json,audit.json,generalization_diagnostic.json}`，再回溯 on-policy donor 与 incumbent-slate；三轮负结果都保留。
3. 最新本地资产为 336 个 selector checkpoint、105 rank、105 count、21 choices；对应的 JSON 冻结清单给出 identity/hash。不要删除重训。
4. 查看 `selection_amendment.md/json`：只改内层选择分母，无交换差值为零不是安全通过；求解未定行采用保守上下界。
5. 查看第 14 节未完成事项。用户已要求 Codex 停止继续研究，交接中的建议不等于新实验已获批准或已经运行。
6. 作者 Word 和 v219 证据候选尚未合并，投稿门户未在本次核查；不要自动提交。

交接时没有留下已知需等待的本地 M3W 训练或评分任务。前序 PID 84281、84483、84642 和最新实验 PID 87216、87394、87503、88022、88323、88427 均有历史终态；此次按进程名检查未发现相关 Python 任务。PID 会复用，不能据此去终止当前同号进程。其他数据湖任务没有被取消或全面审计。
研究总目标未完成；当前不是“投稿准备已完成”。这份交接的作用是让下一位协作者能继续实际工作，而不是从旧成功标题或旧会话记忆重新猜测项目状态。

## 12. 最新完成实验：相对收益与低容量嵌套验证

状态：**已训练、冻结、评分、诊断，主假设失败。** 四组实验 `opendd_advantage_factorial_v1` 完成 260,400 updates。注册、模型选择和外层 choices 均有冻结记录；最后 27 项相关测试通过，不代表全库测试通过。
`opendd_nested_advantage_v1` 的并行五组草案未被覆盖；该路径原四组注册因协议哈希冲突在训练前停止，0 updates，见 `runtime_conflict.json`。两套方案不可混用。

已实施的对照是线性模型 / 48-32 GELU MLP × 绝对 full-grid pass 预测 / 相对 keep 的 switch advantage。共享因果特征、仅训练地点拟合的标准化、固定预算、3 seeds、每模型 400 updates。绝对目标为 BCE；相对目标为 `tanh(score(option)-score(keep))` 对 full-grid 差值 -1/0/1 的平方误差，不是概率。

作者已批准全部验证录像上的净收益：无交换仅贡献零差值，不能算风险通过。126 个内层地点/seed 验证单元中有 26 个没有交换；另有 2 个重复录像单元求解未定，保存 [-1,+1] 差值界，选模型时用保守下界。原 2% 规则、外层 mask、空对照不可估和冻结测试规则未改变。

**关键是嵌套排除，而不是把旧训练包切成两个集合：**

1. 对外层留出地点 H、内层验证地点 V，训练 donor D 的上游模型必须同时排除 H/V/D。
2. 若该上游模型在另一个地点 T 上用 teacher 训练，产生 T 候选的 producer 还须排除 T，即 H/V/D/T。
3. 不能用仅排除 H/D、实际见过 V 的旧 producer 给内层训练数据做特征，再声称 V 是干净验证集。
4. 内层选定模型后，冻结全部外层地点/seed 的选择，再统一评分；不得根据外层结果回调 keep threshold。

实际新增 105 个四重排除 ranker × 700、105 个三重排除 count head × 500、252 个内层选择器 × 400、84 个外层选择器 × 400；复用旧 producer 时按 identity/hash 检查。420 个 teacher cache、441 个 packet cache 均保留本地。来源核验和 126 个内层验证单元、21 个外层选择单元的重放已通过。

保持旧 outer 101 eligible views、780 edits、原 2% 规则及统计单位。若内层只能使用 all-available donor slates、不能复现 outer eligibility，明确记录这个群体差异。主比较仍应涵盖 incumbent、相同 switch mask 的 uniform 和旧 paired selector，不能仅报告最有利 seed 或模型。

| 策略 | full-grid 通过数 | 救回 | 破坏 | 交换数 |
| --- | ---: | ---: | ---: | ---: |
| 保持原分配 | 55 | 0 | 0 | 0 |
| 前序 paired selector | 52 | 10 | 13 | 82 |
| linear absolute | 51 | 7 | 11 | 84 |
| linear advantage | 48 | 7 | 14 | 88 |
| MLP absolute | 50 | 5 | 10 | 87 |
| MLP advantage | 51 | 9 | 13 | 79 |
| 内层验证选出的策略，主比较 | 48 | 7 | 14 | 90 |
| 主策略同 switch-mask uniform 期望 | 48.4955 | 6.0393 | 12.5438 | 90 |
| 事后 oracle，仅诊断 | 100 | 45 | 0 | 不作真实部署 |

统计单位为 7 地点，先平均 3 seeds 再做 3,000 次地点 bootstrap。主策略相对 incumbent、同 mask uniform、前序 paired 的 CI 分别为 `[-0.78740,0]`、`[-0.11035,1.40273]`、`[-0.44994,0]` pp，均不满足预先登记的正下界要求。原风险支持仍有 402 个主策略视图不可估，另有 121 个已知违规，不得把“无交换差值为零”扩写成安全通过。

已有诊断的结论：

- 全部净损失来自 rdb1：主策略通过数 36，incumbent 43；其他地点净差为零。这不等于其他地点没有相互抵消的救回/破坏。
- MLP absolute 在重复训练 donor 上 chosen full-grid rate 为 52.69%，外层 eligible 为 18.81%；相应未校准 sigmoid ECE 从 4.73% 到 43.71%。训练能拟合，但不能迁移。
- MLP advantage 在训练 donor 上救回 696、破坏 0；外层只救回 9、破坏 13。不能把训练拟合当泛化。
- linear advantage 在外层平均预测差为 +0.4762，实际为 -0.0693；换成相对收益目标没有自动解决高估。
- 降低容量也不够：线性绝对头最多 51，仍低于 55。新四组对照不支持“只改 loss 或缩小模型即可修好”的说法。
- 内层 all-available slate 和外层固定 eligible mask 的群体差异仍存在；这是一项待检验解释，不能单凭本轮就认定唯一根因。

最新文件结构：

```text
scripts/run_m3w_opendd_advantage_factorial.py   # register/producers/inner/outer/score
scripts/m3w_advantage_factorial_data.py        # 显式路径与嵌套数据角色
src/world_model/m3w_advantage_factorial.py     # Selector(arm,norm)、loss、choose
scripts/audit_m3w_advantage_factorial.py       # 排除关系、标准化、精确重放
scripts/diagnose_m3w_advantage_factorial.py    # 已冻结后的诊断，不调参
tests/test_m3w_advantage_factorial.py

data/stage_cvpr2027_experiments/opendd_advantage_factorial_v1/
  producers/rank_*.pt / count_*.pt            # 210 权重；相邻 JSON 来源记录
  selectors/*.pt                             # 336 权重，含 step/net/opt/rng/trace
  teachers/*.npz / *.json                     # 420 缓存及来源
  packets/*.npz / *.json                      # 441 缓存及来源
  inner/*.json                               # 63 个排除对×seed 单元
  choices/*.json                             # 21 个外层冻结单元
  events_*.jsonl / heartbeat.json / process.lock
```

复现命令只供 Claude 在需要时使用，本次交接不执行。没有通用 `--resume`，同阶段按 checkpoint 恢复；缺文件会实际训练，不能误称只读：

```bash
cd /Users/yangyue/Downloads/World
export LC_ALL=C OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_advantage_factorial.py register
.venv-pytorch/bin/python scripts/run_m3w_opendd_advantage_factorial.py producers
.venv-pytorch/bin/python scripts/run_m3w_opendd_advantage_factorial.py inner
.venv-pytorch/bin/python scripts/run_m3w_opendd_advantage_factorial.py outer
.venv-pytorch/bin/python scripts/audit_m3w_advantage_factorial.py --pre-score
.venv-pytorch/bin/python scripts/run_m3w_opendd_advantage_factorial.py score
.venv-pytorch/bin/python scripts/audit_m3w_advantage_factorial.py
.venv-pytorch/bin/python scripts/diagnose_m3w_advantage_factorial.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_advantage_factorial.py tests/test_m3w_onpolicy_donors.py tests/test_m3w_incumbent_slate.py
```

主登记 SHA-256 为 `a191473e647c0e602ffad84d77d21bfac9cbfb44ad7abd3b1a78a5b6f735e0ac`。
选择分母修订单独绑定 runner 新哈希，保留原登记；不要删 registration 来绕过不匹配。
全部完整运行有正常 complete 终态，原生 arm64 CPU4、workers0，峰值进程 RSS 约 1.59 GiB。
本轮没有新增 CREATE 作业、外部独立确认或部署放行。新结果尚未进入论文。

上一研究续接中已检查过、此次交接保留的相关先前实验：

| 公开目录，均在 `outputs/publication_readiness_2026_09/` 下 | 已覆盖内容；不能重复宣称首次完成 |
| --- | --- |
| `opendd_candidate_advantage_v1/` | query/candidate 层面的 gain/harm；不是当前整段 allocation + keep 的同一任务 |
| `opendd_slate_rank_v1/` | 成对排序与同 query recipient slate；不能把 pairwise loss 本身写成新方法 |
| `opendd_nested_spline_choice_v2/` | 嵌套校准排除与选择；不是新的 policy-matched 整段 donor 设计 |
| `opendd_credit_value_v2/` | credit/queue timing 与 advantage；预算时序和当前冻结 slate 不同 |

交接后若用户先要求论文或调用说明，应优先处理新请求；本节不是无条件自动启动训练的指令。

## 13. 交接成功的核对标准

Claude Code 应能不依赖旧聊天，回答：当前实验是哪一个、主假设是否通过、代码与权重分别在哪里、如何恢复、什么数据仍未获准用于确认、作者正在改哪个 Word、哪些旧结论不能直接引用。

本文件已覆盖主仓库、所核实的嵌套第三方仓库、远端项目位置、核心调用链、模型/数据/论文资产、历史路线、最新失败证据、恢复命令、用户约束和未实施方案。没有把私有数据或凭据复制进文档，也没有把历史记录伪写为本次新实验。

## 14. 任务移交清单与未完成项

| 任务 | 移交状态 | 接手位置/注意事项 |
| --- | --- | --- |
| 历史 SDD、external、JEPA/Transformer/Hybrid 路线 | 代码和旧报告已有；本次未重跑 | 第 3、4、5 节和三份历史总表；历史正结果受后续来源审计约束 |
| 因果 8/12 与 OpenDD 强对照 | 已有结果和冻结记录 | `outputs/publication_readiness_2026_09/`；不要混合旧 t50 指标 |
| incumbent-slate、on-policy donor、factorial | 三轮均已完成，主假设未通过 | 各自 README/readout/audit，最新见第 12 节 |
| WEPDTOF、Songdo、Athens | 已有正式真实数据 readout；联合外部优势未证实 | 第 5.2 节；固定提案和实际放行分开，缺失标签包络不是 CI |
| DroneCrowd 独立确认 | 未运行；校准 gate 未通过 | 保留 15/30 协议，不为调参打开确认标签 |
| Swiss 等外部方向 | 部分准备/演练，不能算真实成功 | 相应 registration/receipt，先确认数据角色 |
| 后续模型修复 | 未运行、未另行注册 | 可研究 eligible 群体匹配、跨地点可靠性和保守增益；不要事后调整这轮结果 |
| 论文、图表和补充 | v219 独立证据候选已生成；作者 Word 单独维护 | 第 7 节；最新三轮负结果尚未合并，不覆盖作者修改 |
| 匿名复现包 | v219 已有；不是全训练复现包 | `replay.zip` 与中文教程；最新实验需另行决定是否纳入 |
| 作者/门户/利益冲突/AI 声明 | 本次未查看或填写，不确认提交状态 | 由作者核实，不凭聊天猜测，不自动提交 |
| CREATE | 有历史作业及资产位置；本次未查询 | 第 8.2 节；不要复制启动、不要改其他项目 |
| Git 同步 | 交接提交仅含自己的文档、已生成轻量结果与测试补充 | 保留旧暂存，原始数据/权重/缓存只在本机 |
| Stage5C / SMC | 未执行，仍禁止 | 不因交接或旧脚本存在而启动 |

仍缺的科学证据是：学得的联合策略在未参与调参的外部场景上稳定优于强对照，同时满足原定风险规则。
这轮失败不能写成已达到投稿候选质量；也不能把“已同步给 Claude 的文件”写成 Claude 已核验研究结论。
