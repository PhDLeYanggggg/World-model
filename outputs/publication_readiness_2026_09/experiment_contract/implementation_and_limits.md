# M3W 独立实验协议入口与来源约束

2026-09-16。本轮是 `fresh_run` 工程实现与拒绝测试；既有 raw-position cache 为 `cached_verified`。没有新增预测训练、准确性评价、独立确认集或部署提升。

## 解决的问题

历史审计已经表明：文件路径不同不等于录像独立，重新分配 split 也不会消除旧 teacher 的训练暴露。新入口将科学协议、数据角色和所有学习产物的来源显式关联，以便正式训练使用同一套可复查约束，而不是依赖报告中的文字承诺。

实现：`src/evaluation/m3w_experiment_contract.py`。

1. **协议冻结**：hash 覆盖数据角色、观察/预测定义、指标/聚合方式、风险定义、seeds 和 bootstrap 规则。只有带有匹配协议 hash 与决策出处的明确批准记录才能打开入口；改协议后旧批准失效。代码不自动签发批准。
2. **四类用途**：fit、development、calibration、confirmation 必须明确。同一物理场景不能跨用途；同一 fit 场景不能跨 crossfit folds。排除数据不能通过读者接口打开。
3. **历史暴露不洗白**：确认性协议拒绝将 `development_exposed` 或 `unknown` 来源用于独立 calibration/confirmation。探索性协议允许研究重跑，但不得宣称获得独立确认。
4. **缓存与实现身份**：metadata、数组和指定源文件/实现 hash 校验；cache 的录像/场景身份需与 reviewed catalog 一致。已知相同原始文件或点数组不能通过重命名场景跨用途。此处不证明变换、重编号后的所有重复都已识别。
5. **完整依赖链**：模型、预处理、goal prototypes、risk head、policy 和 calibrator 都需声明 fit/selection/calibration 来源及父产物。递归检查父 teacher；不能只检查最外层 selector。OOF 预测排除整个目标 fold，而不只是当前行或录像。
6. **校准前冻结策略族**：读取 calibration 前登记固定产物族及依赖 hash。校准后新增候选、改权重或换配置不会被当作同一次校准继续执行。
7. **最终测试一次性登记**：协议固定 receipt 路径，独占创建；重复新建被拒绝。中断后只允许同一协议和同一产物身份 resume。完成后不能通过该接口再次作为新确认运行。结果文件的 hash 被记录，但文件存在不等于指标正确。
8. **时间定义明确**：只支持精确 raw-frame 或 observation-step 视图。没有精确 t25 标签时拒绝，不用邻近帧代替。当前不批准 seconds 或 metric 声明。

## 本轮实际执行

| 检查 | 实际结果 | 边界 |
| --- | --- | --- |
| 真实开发缓存快照 | 9 个 canonical recordings 的源/metadata/array 身份核对 | 使用既有缓存，没有新转换 |
| 历史用途 | 9/9 存在旧 cache 使用记录，按 development-exposed 保留 | 没有新 untouched holdout |
| 未批准草案正常 preflight | exit code 2；明确拒绝 | 预期安全行为，不是训练卡死 |
| 新协议入口针对性测试 | 25 passed | 合成 fixture 批准不代表真实用户批准 |
| 联合相关回归测试 | 73 passed in 2.68 s | 包含上述 25 项与既有 48 项 |
| 新正式训练 / 独立确认 / calibration | not_run | 科学选择待确认，独立源尚未补足 |

未重跑没有变化的旧全套。此前全套仍是 1,870 passed / 1 个数据湖 fixture failure，不能写全套通过。没有修改无关数据湖工作。

机器记录：`preflight.json`、`unapproved_rejection/preflight.json`。草案：`configs/m3w_independent_experiment.draft.json`。当前所有 recording assignment 都是 `unassigned`，主指标/时域/风险预算没有被脚本擅自填入；easy degradation 上限 0.02 沿用用户已给定边界，不等同于有界 calibration loss 的 tolerance。

## 重要限制

- 这是新训练路径可调用的前置接口，**不是全仓库访问控制**。旧脚本并未全部迁移；绕开接口直接读文件不在这套工程检查的防护范围。后续新训练入口必须实际调用，而不能只在训练后补一张 manifest。
- 当前实现支持严格物理场景隔离的候选设计，不等同于已批准的标准 ETH/UCY sequence-level benchmark。观察/预测长度可对齐公开任务，但不意味着划分和统计问题完全相同；正式采用哪种比较仍待确认。
- hash 证明内容一致，不证明申报真实。`approved_by` 不是身份认证，`lineage_complete` 不是模型未见过数据的密码学证明。人工决策、原始训练日志与数据源审计仍必需。
- 在同一 calibration receipt 内恢复不允许更换候选集合；独立性仍要求人在看到校准结果后不通过重新批准另一个协议继续挑选。软件不能替代科研约束。
- 只读/返回用途正确的 reader 不能证明所有下游函数都忽略 future labels。现有未来破坏测试继续有必要，正式模型也要补输入依赖检查。
- 场景组身份不证明 IID，也不保证跨域交换性。当前六个物理场景组限制仍在；协议入口不创造更多统计样本或有用的风险上界。
- checkpoint/config 冻结不是表现证明。没有新的 ADE/FDE、easy preservation、预测安全或论文贡献结论。

## 复现

```bash
.venv-pytorch/bin/python scripts/check_m3w_experiment_contract.py --report-dir outputs/publication_readiness_2026_09/experiment_contract/unapproved_rejection
.venv-pytorch/bin/python -m pytest tests/test_m3w_experiment_contract.py tests/test_m3w_joint_intervention.py tests/test_m3w_causal_recordings.py tests/test_m3w_recording_lineage.py tests/test_stage44_worldcore.py tests/test_stage42_source_level_ucy_full_waypoint_integration.py -q
```

第一条对当前未批准草案应返回 2。不要为了让它变绿而改 `status`、清空历史暴露、伪造决策出处或复用旧 teacher。`--snapshot-draft` 仅用于在新路径生成尚未选择科学参数的资产草案；它拒绝覆盖已有配置。

正式实现顺序：用户确认主评价协议与探索/确认分工，补足可用且独立的来源，再在 fit 内重训 predictor、产生合法 OOF gain/harm 监督，development 选候选，冻结后 calibration，最终 confirmation。先前讨论的 obs8/pred12 主表与 raw-frame 补充仍只是建议。

下一项可独立推进的工作是核查本地其他外部来源的录像身份、既往使用和使用条件，不先读取其预测结果来挑选 holdout。CREATE 仍待认证与项目路径；没有新远程作业。Stage5C、SMC 继续关闭。
