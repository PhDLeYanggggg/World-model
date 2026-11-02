# 独立风险校准：本地运行、恢复与结果核查

这份说明针对 `european_independent_risk_calibration_v1`，不是旧 SDD
raw-frame t+50 实验，也不是从零训练 M3W 的完整匿名复现包。
当前实验使用已训练的 SOURCE 模型，只校准介入风险约束。

## 1. 先区分三种工作

- **已有训练结果**：九套 SOURCE forecaster/cost 模型及其预处理，使用前校验来源和哈希，属于 `cached_verified`。
- **本轮真实计算**：102 段校准录像的输入、九套模型预测、完整策略网格和后续评分，完成后才标记 `fresh_run`。
- **未运行**：尚未完成的参数选择、最终确认或新的训练，保留 `not_run`，不能用旧分数代替。

模型是否可重复运行，与模型是否有效，是两回事。输入和预测逐项一致，
不代表精度提升、风险通过、可部署或已达到投稿候选质量。

## 2. 环境

在项目根目录运行，使用原生 arm64 环境，不用默认 x86_64 Conda：

```bash
uname -m
.venv-pytorch/bin/python -c 'import platform; assert platform.machine() == "arm64"; print(platform.machine())'
```

当前已核实路径是 Torch CPU，计算线程 4、interop 线程 1，没有 DataLoader
多进程或硬件资源探测。入口会在导入 Torch 前拒绝 macOS x86_64/Rosetta。
这些检查不替代真实运行证据：本轮已实际完成全部冻结模型推理及完整复算，
实际恢复试跑也已完成。具体 PID、时间和内存见 `execution_status.md`。

本实验没有运行 MPS，也没有把 NumPy fallback 当作 Torch 模型成功。
不要在同一已冻结实验中随意改变设备、线程或数值容差后沿用旧复算结论。

## 3. 输入与数据边界

预定规模是全部 102 段录像、12 个 locality，每段最多 256 个过去支持的
查询帧，obs8/pred12/raw stride12。它不是原始录像所有窗口的全量训练。
已构建 481,537 个目标窗口和 801,360 条可见 agent 记录。

只用当前及过去信息构建输入，包括部分历史邻居。目标是否有完整未来，
不影响其过去支持资格。未来位置和未来有效性不能进入推理加载器。
校准目标只通过独立监督/评分接口读取，即使这些文件存放在同一录像目录。

这里“尚未打开校准标签”指尚未构建未来监督窗口、计算误差或读取校准结果。
原始 CSV 解析本身包含各帧轨迹，因此不能说从未解析时间更晚的原始行。
因果输入的要求是：每个查询的实际特征只使用不晚于该查询的观测，并通过
前缀一致性核验；不是把整份录像文件描述成不含未来。

场景数不等于独立样本数；重叠窗口和九套模型的重复视图不能当作独立观测。
一个 locality 占约 86.35% 的目标，因此保留 locality 分组和全部录像。

## 4. 查看正在运行的任务

本轮完整分配启动时的 PID 是 8905；这只是一个有日期的记录，不代表该
PID 永远属于这个任务。核查实时进程的命令与日志，而不是只看旧 heartbeat：

```bash
ps -p 8905 -o pid,etime,%cpu,rss,stat,command
tail -n 5 data/stage_cvpr2027_experiments/european_independent_risk_calibration_v1/allocation_full_v2.log
```

只有确认对应进程仍活着且进度推进，才能称“正在运行”。观察超时不等于
进程结束；旧锁文件或旧 heartbeat 也不证明进程存活。不要因为一次观察
超时就另起任务，也不要因为计算慢而停止或减少预定录像/模型数量。

本轮保留 10GiB 空闲空间、6GiB 私有缓存上限及每阶段 12 小时边界。
若达到真实资源边界，保留检查点并解决资源问题，不把缩小运行冒充完整结果。

## 5. 完整策略分配与恢复

先确认没有同一活任务，且全局 `policy.json` 尚未完成，再运行：

```bash
.venv-pytorch/bin/python scripts/run_m3w_calibration_atomic_allocation.py run
```

**这个入口自动恢复，不加 `--resume`。** 它按查询帧、校准强度和录像保存
进度，重用已完成且哈希一致的网格。不要删除有效的 `grid_prefix_v2.gz`。

新的外层检查点把选择矩阵、已完成前缀和绑定信息一次性原子提交，避免
旧版两个文件更新之间的中断窗口。真实最大录像的受控停止、重新加载、
直接完整运行及原始试跑结果已一致；这不是任意 OS-kill 场景的证明。

同一命令不是强制覆盖工具。完整全局 freeze 已存在时，程序拒绝重新运行。
此时进入下一阶段或核查已有证据，不删除结果制造“fresh_run”。

## 6. 何时才能读取校准目标

完整策略冻结后，可先生成不读取目标标签的选择与求解器统计：

```bash
.venv-pytorch/bin/python scripts/report_m3w_calibration_allocation.py
```

它记录切换数量、各 baseline 选择和求解器回退，不重新求解 MILP，
也不证明预测精度或真实风险通过。

必须先完成全部 918 个录像与模型组合的所有网格并写出全局 freeze：

```bash
.venv-pytorch/bin/python scripts/run_m3w_independent_risk_calibration.py labels --resume
```

标签入口会检查完整 freeze；缺少它不能执行后续评分。`--resume` 只恢复
本阶段已写入且绑定一致的校准标签，不允许使用 test/confirmation 标签。
整个标签阶段已完成时，同样不要反复运行覆盖它。

SOURCE、model selection、risk calibration、final confirmation 是不同角色。
本次只按已有登记打开风险校准用途；原角色清单与旧训练接口不改动。
最终确认集和 DroneCrowd 不因校准程序跑完而自动开放。

## 7. 评分与验证

在标签阶段完整完成后依次执行尚未完成的步骤：

```bash
.venv-pytorch/bin/python scripts/run_m3w_independent_risk_calibration.py eval
.venv-pytorch/bin/python scripts/run_m3w_independent_risk_calibration.py verify
.venv-pytorch/bin/python scripts/report_m3w_calibration_failure_taxonomy.py
.venv-pytorch/bin/python scripts/report_m3w_calibration_paired_effects.py
.venv-pytorch/bin/python scripts/report_m3w_calibration_cost_attribution.py
.venv-pytorch/bin/python scripts/report_m3w_independent_risk_calibration.py
.venv-pytorch/bin/python scripts/plot_m3w_independent_risk_calibration.py
```

每个完成标记都是不可覆盖的阶段记录。已有同版本有效验证可直接复用；
不要把遇到“completed phase immutable”误当成模型崩溃，或为了重跑删掉证据。

主要读取：

- `summary.json`：固定网格、共同校准参数和原始记录级风险结果。
- `readout.md`：可读结果及其数据角色、指标和限制。
- `verify.json`：评分、可行性与逐查询切换数量核验的实际范围；不等于再次优化所有 MILP。
- `risk_failure_taxonomy.md`：真实伤害、缺失未来边界、支持不足等原因；分类不是另一次调参。
- `paired_calibration_effects.md`：直接成对计算未知未来的预测差异；不是两个各自下界相减，也不参与选参。
- `selected_cost_attribution.md`：明确在读数后加入的描述性成本归因，区分伤害低估和参考误差高估；不重训、不修改本轮参数，不声称事先登记。
- `gates.md`：实际通过与未通过项，不能以程序成功退出替代科学 gate。

结果图只读取公开汇总，不重新选择参数。私有 PNG 预览和公开 SVG 来自同一
张 Matplotlib 图；图中保留全部网格、失败和不可估结果。预测或风险上的正值
仍需结合完整 gate 阅读，不能以其中一张正收益图替代部署证据。

校准参数只从预先固定的 13 个强度中选择，模型、排序目标和 2% 要求不变。
所有必需录像视图都要满足原始条件；分母不可估不能填成安全的零。
若没有任何共同参数满足要求，报告校准不支持，不把全回退说成学习成功。

校准数据用于选参数，所以这里的效用和 locality bootstrap 仍是条件性
校准读数，不是最终独立效果，也不是一般人群的 2% 风险证书。

## 8. loss、测试和复现范围

本轮不更新神经网络参数，因此没有新的训练 loss 曲线。已有 SOURCE
训练 loss 与检查点属于原训练实验；不能把本轮推理或 MILP 优化叫新训练。

针对本轮接口和公开证据的检查：

```bash
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_calibration_atomic_progress.py tests/test_m3w_independent_risk_calibration.py tests/test_m3w_independent_calibration_evidence.py tests/test_m3w_calibration_failure_taxonomy.py tests/test_m3w_model_selection_inputs.py tests/test_m3w_selection_feature_bridge.py tests/test_m3w_frame_action_allocation.py
```

针对性检查不等于整个 `tests` 仓库通过。全套未运行时必须单独标明。
完整评分和 `verify` 都结束后，还可用公开充分统计量重新核对全部 104 组
地点级点估计与区间，以及 39 个配对比较；不加载私有轨迹、不训练模型：

```bash
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_completed_calibration_evidence.py
```

它保留原程序按录像首次出现的地点顺序，因此固定种子 bootstrap 能逐值
匹配；不能把 JSON 排序后的键顺序偷偷换进去后声称逐值一致。这仍只是
数值报告复现，不是新的独立数据验证或完整冷启动训练复现。
本地存在的私有原始数据、cache 和模型权重不会随 Git 分发；全新机器缺少
这些资产时，这份阶段操作说明不能冒充已验证的冷启动完整训练复现包。

## 9. CREATE 的使用边界

本轮资源试跑支持本地执行，没有提交 CREATE 任务。旧实验的 CREATE
运行或只读检查记录，不能当作当前排队状态、操作系统迁移兼容性或新作业存活证据。

需要迁移时，先沿用已有授权 handoff 核查队列与环境。仓库的
`python -m scripts.check_m3w_create_queue_verified --receipt <新的私有回执路径>`
是既有只读核查接口，Python 仍使用 `.venv-pytorch/bin/python`；需要本机已有
授权 SSH/handoff 资产。该旧入口应以模块形式启动，直接运行文件可能找不到
`scripts` 包。`--help` 能运行只证明入口可加载，不证明远程连接成功。不要为通过访问
而修改认证或主机校验，也不要碰 simulation 项目的受保护环境。

`prepare_m3w_create_runtime.py` 等旧提交入口绑定的是旧实验，不能直接用于
提交这次校准。当前校准的 CREATE 提交脚本和迁移后环境验证尚未完成，
因此不能宣称已完成本实验的 HPC 复现。实际迁移必须记录作业 ID、环境、
资源、日志、输出和恢复方式；访问失败时先保留现有任务，不推断它已结束。

## 10. 结果声明与 Git

当前单位是 local image pixel/raw frame，标签是 released detector/tracker
silver，不是 human gold。部分未来上的 ADE 不等于完整十二步 ADE；
整行未知未来的边界也不恢复部分轨迹中缺失的步。不可声称秒、米、true 3D
或 foundation success，Stage5C 和 SMC 继续关闭。

只提交自己的代码、配置、报告、轻量指标与衍生统计 SVG。原始数据、特征、
历史、latent cache、选择矩阵、模型权重、视频图像和 `.venv-pytorch` 不提交。
已有无关 staged/untracked 内容不回退、不打包进本任务提交。
