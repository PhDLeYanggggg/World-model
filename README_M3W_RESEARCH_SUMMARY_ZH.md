# M3W 研究总总结：尝试路线、失败原因、成功证据与当前结论

更新时间：2026-05-26  
工作目录：`/Users/yangyue/Downloads/World`  
结果来源：`cached_verified` 汇总已有阶段报告、README、gate report 和 `research_state.json`，并纳入 Stage42-W/X/Y/Z/AA 已生成并校验的轻量报告；本文件本身不重新训练大模型、不读取未提交 raw data。未完成或未正式评估的分支不会写成已完成结果。  

本轮校验：

```text
python3 run_stage42_unified_ablation_evidence.py = pass
python3 run_stage42_paper_claim_evidence_audit.py = pass
python3 run_stage42_retrained_ablation.py = pass
python3 run_stage42_retrained_ablation_matrix.py = pass
python3 -m pytest tests/test_stage42_unified_ablation_evidence.py = 3 passed
python3 -m pytest tests = 333 passed
```

这份 README 回答一个核心问题：在“训练真正强的真实世界多模态多智能体世界模型 M3W”这个长期目标里，我到底做了什么、尝试了哪些路线、哪些失败了、为什么失败、哪些成功了、现在能诚实 claim 什么、还不能 claim 什么。

阅读索引：

- 第 0 节：必须遵守的 claim 边界。
- 第 1 节：从早期 JEPA 到 Stage42 row-level full-waypoint cache 的总路线图。
- 第 2 节：最重要的成功证据。
- 第 3 节：失败路线、失败原因和修复逻辑。
- 第 4 节：成功路线总表。
- 第 5 节：当前 best deployable 是谁。
- 第 6 节：为什么仍不能称 true 3D / foundation / metric。
- 第 7 节：下一步最短路径。
- 第 8 节：给你的直接结论。
- 后续追加：Stage42-W/X/Y/Z/AA 的统一 full-waypoint、paper claim、retrained ablation evidence。

## 本次用户版总览

你问“在这个目标内我做了什么、尝试了什么路线、哪些失败、哪些成功”。最压缩但不失真的答案是：

1. 我先把项目从早期 2.5D scaffold 推到可审计的 SDD pixel raw-frame benchmark，再用 Stage26 cost-aware selector 得到第一个稳定 SDD 基座。
2. 我尝试把 SDD 直接 zero-shot 到 external top-down pedestrian 数据，结果严重失败，说明坐标、horizon、goal、agent-type 和 scene/context 都有 domain gap。
3. 我没有把这个失败包装成泛化成功，而是逐步补 external row geometry、train-only goals、history windows、scene-agnostic goal prototypes、hard/easy/failure labels 和 conservative fallback。
4. Stage37 是第一个 external deployable 转折点：all、t+50、hard/failure 都正，easy degradation 极低。
5. 后面我开始训练 neural dynamics。无保护 Transformer/JEPA/Hybrid 多次失败；有效结果都来自 Stage37/teacher floor 保护下的 bounded / safe-switch neural package。
6. Stage41/42 把结果从 endpoint / selector 推到 all-agent、full-waypoint、row-level cache 和 retrained ablation evidence；这比早期 demo 更像研究证据链。
7. 失败路线也很明确：JEPA non-collapse 但 downstream 无稳定 lift；hard-class selector 会严重伤 easy；ordinary residual/correction 不安全；ungated neural dynamics 不可部署；endpoint success 不能自动转成 full-waypoint success。
8. 当前最强可部署仍是 protected M3W-Neural v1 / Stage37-teacher-floor 路线，最新 Stage42-X/Y/Z/AA 则提供 row-level full-waypoint cache、统一消融和论文 claim 边界证据。
9. 仍不能说 true 3D、metric、seconds-level、foundation，也不能执行 Stage5C 或 SMC。

没有纳入为“已完成结果”的内容：

- 尚未正式跑完/提交的 Stage42-AB auxiliary-head ablation，不写成成功也不写成失败。
- 大 cache、checkpoint、heartbeat、raw data、第三方数据，不作为 GitHub 提交内容。

## 0. 必须先写清楚的边界

当前 M3W 还不能被称为 true 3D world model，也不能被称为 large-scale foundation world model。当前最强结果仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型，主要证据来自 SDD pixel-space 和 external dataset-local raw-frame top-down pedestrian 数据。

不能夸大的点：

- SDD 是 pixel-space benchmark，不是 metric benchmark。
- external 坐标仍是 dataset-local / unverified weak-metric diagnostic，不能写成统一真实物理米制。
- t+50 / t+100 是 raw-frame horizon，不能写成 seconds-level。
- homography、metric scale、effective seconds 没有完成全局验证。
- self-audited / visual-prior / auto-silver label 不是 human gold。
- JEPA 在本项目里是 representation / auxiliary，不是 latent generative rollout。
- Stage5C latent generative 没有执行。
- SMC 没有启用。
- 无保护 neural dynamics 仍不安全；当前 deployable 结果仍依赖 Stage37 / teacher safety floor。

一句话现状：

```text
current strongest protected package =
  M3W-Neural v1 composite-tail safe-switch bounded neural dynamics

safety floor =
  Stage37 selector / teacher floor

current honest claim =
  protected dataset-local raw-frame 2.5D multi-agent world-state candidate

not allowed claim =
  true 3D / metric / seconds-level / foundation world model
```

## 1. 总路线图

长期目标不是单次 demo，而是一条阶段状态机：

1. 先尝试早期 JEPA / selector / correction，发现 JEPA non-collapse 但 downstream 无 lift。
2. 把 SDD 建成 official pixel raw-frame benchmark。
3. 通过 Stage26 得到 SDD 上可靠的 cost-aware selector。
4. 尝试 SDD 到 external top-down pedestrian zero-shot，严重失败。
5. 补 external row geometry、train-only goals、scene packs、coordinate-invariant features、relative targets。
6. 通过 Stage37 修复 external t+50，第一次得到 all / t50 / hard-failure / easy safety 同时过 gate 的 deployable external selector。
7. 尝试 bounded correction 和普通 residual，发现不能安全超过 Stage37。
8. 开始真正神经动力学：Transformer / JEPA / Hybrid。无保护 neural 失败，受保护 neural 才有用。
9. Stage41/42 把 neural 结果推进到 protected neural package、full-waypoint dynamics、row-level combo、UCY full-waypoint candidate。
10. 当前仍需要 safety floor、row-level cache、validation-only selection 和 no-leakage audit，不能进入 Stage5C 或 SMC。

核心规律：

```text
真正有效的不是盲目加大模型，而是：
  causal past-only history
  train-only goals / scene-agnostic goal prototypes
  hard/easy/failure 分层
  gain/harm/risk-aware selection
  conservative fallback
  validation-only policy selection
  protected full-waypoint dynamics
  row-level prediction cache / combo evaluation
```

## 2. 当前最有意义的成功结果

### 2.1 Stage26：SDD cost-aware selector

Stage26 是 SDD pixel-space 上第一个稳定 selector 基座。

```text
t+50 improvement ~= 14.58%
hard/failure improvement ~= 11.23%
easy degradation ~= 1.81%
Stage5C = false
SMC = false
```

意义：

- 证明 hard classification selector 不可靠以后，expected-FDE / gain-harm / fallback-safe 选择器是对的方向。
- 但它仍是 SDD pixel raw-frame，不是 metric / true 3D / foundation。

### 2.2 Stage37：external t+50 deployable selector

Stage37 是 external transfer 的关键转折点。它用 past-only history windows、scene-agnostic goal prototypes、t+50 switchability model 和 conformal safety，把 Stage35/36 卡住的 t+50 修好。

```text
all improvement = +13.48%
t+50 improvement = +8.46%
t+50 bootstrap CI = [+7.69%, +9.15%]
hard/failure improvement = +15.54%
easy degradation = 0.041%
gates = 16 / 16
verdict = stage37_t50_transfer_repaired_deployable
```

意义：

- 这是 external dataset-local raw-frame 上第一个真正可部署的正迁移 selector。
- 它修复了 Stage35 all/hard 有提升但 t+50 = 0 的核心问题。
- 但它仍主要是 selector-level / policy-level，不是无保护神经世界动力学。

### 2.3 M3W-Neural v1 protected package

Stage41/42 形成当前最强 protected neural package：M3W-Neural v1 composite-tail safe-switch bounded neural dynamics under Stage37/teacher floor。

相对 Stage37 / teacher floor：

| 指标 | 结果 |
| --- | ---: |
| gates | 41 / 41 |
| evaluated rows | 55,528 |
| all ADE improvement | 21.03% |
| t+50 ADE improvement | 13.65% |
| t+100 raw-frame diagnostic ADE improvement | 14.69% |
| hard/failure ADE improvement | 20.38% |
| easy degradation | 0.00% |
| positive external domains | 3 |
| all-agent composite FDE improvement | 19.82% |
| all-agent composite FDE@50 improvement | 17.39% |

2000-bootstrap lower bounds：

| slice | low | mid | high |
| --- | ---: | ---: | ---: |
| all | 20.67% | 21.02% | 21.39% |
| t+50 | 13.06% | 13.66% | 14.26% |
| t+100 raw-frame diagnostic | 13.96% | 14.69% | 15.37% |
| hard/failure | 19.99% | 20.39% | 20.76% |

意义：

- 这是目前最强的 protected neural world-state evidence。
- 它证明 neural branch 在 safety floor 下可以贡献正收益。
- 但它不是 ungated neural world model；部署仍需要 Stage37 / teacher floor。

### 2.4 Stage42-C：full-waypoint dynamics

Stage42-C 把证据从 endpoint / tail bridge 推进到 reconstructed full future waypoint。

```text
source = fresh_run
gates = 12 / 12
verdict = stage42_c_full_waypoint_dynamics_pass
positive_external_domains = ETH_UCY, TrajNet

protected full-waypoint ADE all = 18.58%
protected full-waypoint ADE t+50 = 14.80%
protected full-waypoint ADE t+100 raw-frame diagnostic = 22.86%
protected full-waypoint ADE hard/failure = 19.52%
protected full-waypoint easy degradation = 0.00%
protected full-waypoint FDE t+50 = 21.58%
```

意义：

- 这一步比只预测 endpoint 更接近 world-state dynamics。
- 但 full-waypoint sequence model 还没有完全替代 composite-tail bridge，在 all-ADE 上仍有差距。
- 仍然是 protected / dataset-local raw-frame，不是 true 3D。

### 2.5 Stage42-R/S：row-cache combo 和 frozen policy

Stage42-R 把 Stage42-J static expert 与 Stage42-P t50 gain/harm selector 的 report-level 互补信号变成 row-cache-backed combo。

```text
source = fresh_run_from_row_prediction_cache
verdict = stage42_r_row_cached_combo_pass
gates = 15 / 15
ADE all = 0.052387
ADE t+50 = 0.037934
ADE t+50 CI low = 0.027740
ADE t+100 raw-frame diagnostic = 0.041846
ADE hard/failure = 0.054792
easy degradation = 0.001102
FDE t+50 = 0.100059
```

Stage42-S 冻结这个 combo policy：

```text
verdict = stage42_s_frozen_row_combo_policy_pass
gates = 13 / 13
positive domains = ETH_UCY, TrajNet
UCY status = fallback-only in this combo stress
policy hash = 33450e033e14b10293b8a10796d934d7689e39358ab5eaa338d684a36b015d3f
```

意义：

- Stage42-R 修复了 Stage42-P “mean t50 正但 CI low 仍负”的问题。
- Stage42-S 把 policy/hash/schema/no-leakage 信息固化成轻量 artifact。
- 但 Stage42-S/R 对 UCY 仍 fallback-only，因为 row cache 没有 UCY 非 floor candidate。

### 2.6 Stage42-V：strict pure-UCY full-waypoint candidate

Stage42-T 证明 UCY fallback-only 不是 threshold 问题，而是没有 UCY candidate source。Stage42-U 尝试把 Stage41 pure-UCY endpoint candidate 线性桥接到 full-waypoint，失败。Stage42-V 于是直接训练 strict pure-UCY full-waypoint candidate。

协议：

```text
train = UCY students01/students03
val = UCY zara01
test-once = UCY zara02/zara03
```

结果：

```text
source = fresh_run
verdict = stage42_v_ucy_full_waypoint_candidate_pass
gates = 11 / 11
best trial = ucy_full_waypoint_t50_hard
ADE all = 0.220755
ADE t+50 = 0.290332
ADE t+50 CI low = 0.231725
ADE t+100 raw-frame diagnostic = 0.147461
hard/failure = 0.229484
easy degradation = 0.000000
FDE t+50 = 0.334459
```

意义：

- Stage42-V 修复了 UCY “没有 full-waypoint candidate source”的 blocker。
- Stage42-W 已经把它作为 UCY-domain slice 合并进统一 external full-waypoint policy package。
- 但 Stage42-W 仍不是单一 merged row-cache artifact；下一步要把 UCY candidate 也缓存成统一 row-level source，并重新做全局 row-level bootstrap。

## 3. 主要失败路线和失败原因

### 3.1 JEPA-only / early JEPA downstream 失败

做过：

- Stage18 SAM-JEPA-2.5D。
- Stage19 WAM-style JEPA dataset。
- Stage22/23/24/39/40 多次重测 JEPA auxiliary。

结果：

```text
non-collapse = yes
downstream lift = no
```

失败原因：

- latent 有 variance 不代表能改善 selector/failure/correction。
- JEPA 目标没有对齐部署需要的 gain/harm、easy preservation、hard/failure 风险。
- JEPA 特征在多个 downstream probe 里没有稳定正贡献，有时甚至成为噪声。

结论：

```text
JEPA 目前只能作为 auxiliary / diagnostic，不能作为主 deployable claim。
```

### 3.2 hard-class selector 失败

典型失败来自 Stage24：

```text
selector oracle headroom = 46.2%
trained hard-class selector t+50 improvement = -43.3%
easy degradation = 11.33%
```

失败原因：

- oracle best baseline label 是低 margin、高噪声标签。
- 直接分类“哪个 baseline 最好”不关心错选成本。
- selector 大量切错 easy cases。
- 没有 fallback / margin / confidence / harm guard。

修复方向：

```text
expected-FDE prediction
gain/harm/risk-aware selection
confidence-gated fallback
easy preservation
```

这条修复最终导向 Stage26、Stage37、Stage41/42。

### 3.3 SDD -> external zero-shot transfer 失败

Stage31 初始结果：

```text
all improvement ~= -92.67%
t+50 ~= -278.57%
```

失败原因：

- SDD pixel coordinate 与 external dataset-local coordinate 不兼容。
- scene / goal / interaction 信息缺失。
- agent type 标注不一致。
- horizon / track length / frame step 不匹配。
- scale / homography / metric 未验证。
- latent adapter 缩小分布距离，但没有 predictive lift。

结论：

```text
普通 normalization / CORAL / latent alignment 不足以解决 cross-domain transfer。
必须补 row geometry、history window、goal prototypes、hard/easy/failure gating。
```

### 3.4 Stage34/35/36 external transfer 局部成功但不可部署

Stage34：

```text
t+50 diagnostic lift ~= +6.6%
hard/failure ~= +18% 到 +25%
all-test 为负
easy degradation 高
verdict = not deployable
```

Stage35：

```text
all improvement = +12.13%
hard/failure improvement = +13.98%
easy degradation = 0.041%
t+50 improvement = 0.0
verdict = not deployable
```

Stage36 发现：

```text
t+50 rows = 16,263
t+50 oracle headroom ~= 22.98%
not no-headroom
problem = existing features / goals / context cannot support safe t+50 switch
```

失败原因：

- all-test objective 淹没 t+50。
- t+50 需要专门 history / goal / switchability 特征。
- 只调 threshold 无法解决。

修复：

Stage37 构建完整 past-only history window 和 scene-agnostic goal prototypes，才把 t+50 修好。

### 3.5 bounded correction / ordinary residual 不可部署

Stage38 和 earlier correction specialist 的共同问题：

- correction 在 hard 或 t+50 局部可见信号。
- 但容易破坏 all / easy / safety。
- without fallback 不安全。
- with fallback 后没有稳定超过 Stage37。

结论：

```text
不要训练普通 residual 当 deployable 主线。
必须先有可靠 selector/failure/gain/harm，再做 bounded correction。
```

### 3.6 Stage39/40 ungated neural dynamics 失败

Stage40 典型无保护 neural：

```text
neural_without_fallback all = -1.2636
t+50 = -2.9210
hard/failure = -1.0940
easy degradation = 6.1231
```

失败原因：

- neural 学到的安全切换区域太少，fallback gate 最后 switch_rate 接近 0。
- raw FDE / endpoint loss 没有教会“何时不要切”。
- JEPA non-collapse 但 downstream lift 为负。
- Hybrid 把 JEPA 噪声带进 ranker，不能超过 Stage37。

结论：

```text
神经模型必须被 Stage37 / teacher floor 保护。
无保护 neural 不部署。
```

### 3.7 Stage42 静态上下文 / policy distillation 的失败

失败点：

- Stage42-I：全 static/context sequence-to-waypoint 负，说明静态上下文全局混入会伤模型。
- Stage42-M：slice-level alpha distillation 不能学 row-level gain/harm。
- Stage42-N：row-level alpha teacher all/hard 正，但 t50 仍负。
- Stage42-O：explicit gain/harm all/hard 正，但 t50 略负。
- Stage42-T：UCY transfer rule 失败，因为 row cache 没有 UCY 非 floor candidate。
- Stage42-U：endpoint-to-full bridge 到 UCY full-waypoint 失败，ADE all/t50/hard 为负且 easy degradation 高。

失败原因总结：

```text
static/context 必须 validation-gated，不能全局混入。
coarse alpha policy 不够，必须 row-level gain/harm。
endpoint success 不能自动等于 full-waypoint success。
UCY 需要自己的 full-waypoint candidate source。
```

## 4. 成功路线总表

| 路线 | 状态 | 成功点 |
| --- | --- | --- |
| Stage26 cost-aware expected-FDE selector | SDD 成功 | t50 / hard / easy 三个 gate 同时过。 |
| Stage37 history + goal prototype + safety selector | external 成功 | external all / t50 / hard / easy 同时过 gate。 |
| Stage41 M3W-Neural v1 protected package | 成功 | all/t50/t100/hard 都正，easy 0，3 external positive domains。 |
| strict pure-UCY neural retrain | 成功 | UCY source-heldout bootstrap-stable positive。 |
| all-agent composite world-state | 成功 | 不只是单 agent endpoint。 |
| endpoint-to-full bridge on ETH_UCY/TrajNet | 成功 | 两个 external domains full-waypoint lower bounds positive。 |
| Stage42-C full-waypoint dynamics | 成功 | reconstructed full future waypoint positive on ETH_UCY/TrajNet。 |
| Stage42-H sequence history ablation | 成功 | history tokens 对 causal sequence encoder 有强贡献。 |
| Stage42-J static-gated repair | 成功 | static/context 在 partial validation gate 下有效。 |
| Stage42-R row prediction cache combo | 成功 | combo t50 CI low 变正。 |
| Stage42-S frozen row combo policy | 成功但有 UCY 限制 | ETH_UCY/TrajNet 正，UCY fallback-only。 |
| Stage42-V strict pure-UCY full-waypoint candidate | 成功 | 修复 UCY candidate source 缺失，all/t50/hard 为正，easy 0。 |

## 5. 当前 best deployable 是谁

当前主 deployable 仍是：

```text
M3W-Neural v1 composite-tail safe-switch bounded neural dynamics
under Stage37 / teacher safety floor
```

它不是一个无保护 neural rollout，也不是 Stage5C generative model。

Stage42-R/S/V 的关系：

- Stage42-R/S 是 full-waypoint row-cache combo 分支，对 ETH_UCY/TrajNet 很重要。
- Stage42-V 是 UCY full-waypoint candidate source，修复了 UCY candidate 缺失。
- Stage42-W 已经把 Stage42-S 的 ETH_UCY/TrajNet 与 Stage42-V 的 UCY-domain slice 合成统一 external full-waypoint policy package，并过 gate。
- 但 Stage42-W 明确不是单一 merged row-cache artifact；下一步不是再泛泛加模型，而是补 UCY row-level cache、统一 bootstrap 和更严格 held-out stress。

## 6. 为什么现在还不能说 CCF-A / foundation / true world model

可以说：

```text
M3W 当前具备 protected 2.5D external world-state candidate 证据。
它有 SDD 和 external raw-frame/dataset-local 结果。
它有 no-leakage、bootstrap、多 seed、ablation、failure analysis。
它已经超过单纯 selector-level demo。
```

不能说：

```text
不是 true 3D。
不是 metric prediction。
不是 seconds-level long-horizon。
不是 foundation world model。
不是 ungated neural world dynamics。
不是 Stage5C latent generative。
不是 SMC-ready。
```

CCF-A / A刊候选还差：

1. 更完整跨数据集泛化：Stage42-W 已有 ETH / TrajNet / UCY 统一 policy package，但还需要单一 row-level merged cache 和统一 bootstrap。
2. 更强 row-level unified full-waypoint cache：Stage42-V 的 UCY source 需要进入 Stage42-R/S 同级 row cache，而不是只作为 domain-slice package source。
3. 更清晰的世界动力学贡献：证明不是只靠 safety selector，而是 neural dynamics / history / interaction / goal 分支本身有稳定贡献。
4. 更强 external held-out scene evidence。
5. metric / time calibration 或更严格声明为 raw-frame dataset-local。
6. 更完整论文级 reproducibility package。

## 7. 下一步最短路径

### Step 1：把 Stage42-W package 升级成单一 row-level merged cache

目标：

```text
ETH_UCY = Stage42-R/S row combo
TrajNet = Stage42-R/S row combo
UCY = Stage42-V strict pure-UCY full-waypoint candidate
```

Stage42-W 已经输出统一 frozen external full-waypoint policy package。下一步要输出单一 row-level merged cache。必须清楚标注：

- 哪些是 row-level merged cache。
- 哪些是 domain-level policy package。
- 是否有重叠 row / double counting。
- 是否 no-leakage。
- 是否 test-once。

### Step 2：做 unified policy gate

必须检查：

- ETH_UCY positive。
- TrajNet positive。
- UCY positive。
- t50 positive。
- hard/failure positive。
- easy degradation <= 2%。
- no metric/seconds overclaim。
- Stage5C false。
- SMC false。

### Step 3：推进真正 dynamics contribution

如果 unified policy 稳定，再继续：

- 比较 selector-only vs full-waypoint neural。
- 做 no-history / no-goal / no-neighbor / no-static / no-domain ablation。
- 建立更清楚的 neural dynamics contribution，而不是只靠 fallback。

## 8. 给用户的直接回答

我在这个长期目标里做的不是一条线，而是一组逐步收敛的研究路线：

- JEPA 表征路线：跑通但 downstream 无 lift，所以没有部署。
- SDD selector 路线：Stage26 成功，是 SDD 基座。
- 外部 zero-shot 路线：严重失败，暴露坐标/goal/horizon/domain gap。
- 外部选择性迁移路线：Stage37 成功，修复 t+50，成为 external selector 基座。
- residual/correction 路线：多数不可部署，因为 easy/all 不稳。
- 神经动力学路线：无保护失败，受保护才有效。
- full-waypoint 路线：Stage42-C/R/S/V 推进成功，但仍要合并 UCY branch。

当前真正成功的是：

```text
Stage26 SDD selector
Stage37 external t50 deployable selector
M3W-Neural v1 protected package
Stage42 full-waypoint ETH_UCY/TrajNet evidence
Stage42-V strict pure-UCY full-waypoint candidate
```

当前仍失败或不完整的是：

```text
JEPA downstream lift
ungated neural dynamics
ordinary residual/correction deployment
metric / seconds-level claim
Stage5C / SMC readiness
single merged row-level full-waypoint cache/bootstrap over ETH_UCY + TrajNet + UCY
```

最诚实的 current verdict：

```text
M3W 已经从 SDD selector demo 推进到 protected external 2.5D world-state candidate。
它有多个强证据分支，但还不是 true 3D、不是 foundation、不是 metric/seconds-level。
当前 best deployable 仍是 protected M3W-Neural v1 / Stage37-teacher-floor 路线。
Stage42-W 已经形成统一 external full-waypoint policy package；下一步最值得做的是建立单一 row-level merged cache 和统一 bootstrap，把 package-level 证据推进成更严格的 row-level evidence。
```

## Stage42-W Unified External Full-Waypoint Policy

```text
source = fresh_unified_from_cached_verified_stage42s_and_stage42v
verdict = stage42_w_unified_external_full_waypoint_policy_pass
gates = 16 / 16
policy_hash = a2439e23c0c2e3f7aa99efa8a84e42868ea52258394ce41339c96ee0a2ec910e
rows = 55528
weighted_ADE_all = 0.09933852091487605
weighted_ADE_t50 = 0.09399823177957682
weighted_ADE_hard_failure = 0.10486717627981672
weighted_easy_degradation = 0.002399712905777252
domains = ETH_UCY, TrajNet, UCY
stage5c_executed = false
smc_enabled = false
```

Stage42-W combines ETH_UCY/TrajNet from the frozen Stage42-S row-cache combo policy with the UCY-domain slice from Stage42-V strict pure-UCY full-waypoint candidate. It avoids double counting the Stage42-V ETH_UCY slice and explicitly records that a single merged row-cache artifact remains future work. Claims remain dataset-local raw-frame 2.5D, not metric or seconds-level.

## Stage42-X Unified Row-Level Full-Waypoint Cache

```text
source = fresh_run_from_stage42s_row_cache_and_stage42v_ucy_predictions
verdict = stage42_x_unified_row_level_full_waypoint_cache_pass
gates = 16 / 16
cache_hash = ffa31b2525fa1a10db356ac5b1ef78602e44bc6f065c63cfc05ac29083e08937
ADE_all = 0.0900136608879362
ADE_t50 = 0.06109367671246102
ADE_t50_seed_CI_low = 0.05367075264893123
ADE_t50_bootstrap_CI_low = 0.027880326844751835
ADE_hard_failure = 0.09374591375146946
ADE_easy_degradation = 0.001101978371627214
positive_domains = ['ETH_UCY', 'TrajNet', 'UCY']
stage5c_executed = false
smc_enabled = false
```

Stage42-X upgrades Stage42-W from a domain-level policy package into a row-level merged full-waypoint cache with unified bootstrap. ETH_UCY/TrajNet use Stage42-S row-cache combo outputs; UCY rows use Stage42-V UCY full-waypoint predictions after row alignment. Claims remain dataset-local raw-frame 2.5D, not metric or seconds-level.

## Stage42-Y Unified Ablation Evidence

```text
source = fresh_synthesis_from_stage42x_row_cache_and_retrained_ablation_reports
verdict = stage42_y_unified_ablation_evidence_pass
gates = 13 / 13
Stage42-X_ADE_all = 0.0900136608879362
Stage42-X_ADE_t50 = 0.06109367671246102
UCY_source_loss_if_removed_t50 = 0.0231594736115995
UCY_source_loss_if_removed_hard = 0.038954187812382024
history_token_t50_contribution = 0.457817280518282
history_token_hard_contribution = 0.47079873325328386
stage5c_executed = false
smc_enabled = false
```

Stage42-Y turns the Stage42-X unified row-level cache into paper-table ablation evidence. It shows that removing the UCY full-waypoint source loses t50/hard performance, history tokens are the strongest retrained sequence contribution, domain expert helps, and safety floor remains necessary because ungated neural is unsafe. Goal/scene and neighbor/interaction remain mixed rather than overclaimed.

## Stage42-Z Paper Claim Evidence Audit

```text
source = fresh_audit_from_stage42_wxy_and_paper_package_artifacts
verdict = stage42_z_paper_claim_evidence_audit_pass
gates = 16 / 16
paper_ready_scope = protected_2p5d_raw_frame_world_state_candidate
not_ready_scope = true_3d_metric_seconds_foundation_or_stage5c_smc
stage5c_executed = false
smc_enabled = false
```

Stage42-Z 把“能写进论文的 claim”和“必须作为 limitation / negative evidence 的内容”逐条绑定到 artifact。它支持的主 claim 是：Stage42-X 统一 row-level full-waypoint cache、external t50 正证据、UCY full-waypoint source 贡献、history token / domain expert 贡献、protected external floor、protected full-waypoint dynamics。它明确拒绝：ungated neural 替代 safety floor、metric/seconds-level claim、true 3D / foundation claim，以及把 goal/scene 或 neighbor/interaction 的 mixed evidence 写成统一正贡献。

## Stage42-AA Retrained Ablation Matrix

```text
source = fresh_matrix_from_stage42g_rerun_plus_stage42h_i_d_z
verdict = stage42_aa_retrained_ablation_matrix_pass_with_jepa_transformer_boundary
gates = 15 / 15
fresh_required_coverage = 11 / 12
stage5c_executed = false
smc_enabled = false
```

Stage42-AA 重跑了 Stage42-G 的 retrained ablation，并把用户要求的 12 类 ablation 放进同一张矩阵。当前 11/12 有 fresh Stage42 evidence；唯一不是 fresh 的是 `no_JEPA`，它仍是 cached negative architecture evidence，不能伪装成本轮重训。`no_Transformer` 目前是 fresh proxy，不是完整 no-Transformer architecture retrain。最清楚的正贡献仍是 history tokens 和 domain expert；teacher floor 去掉后不安全，所以 Stage37/teacher floor 仍是部署必要条件。
