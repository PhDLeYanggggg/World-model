# M3W 当前证据稿复现教程

更新：2026-09-22。本文针对观察 8 步、预测 12 步、SDD annotation-pixel 的当前
开发协议，不是早期 raw-frame t+50 selector 的部署说明。训练、选择、校准、最终
确认的用途不能互换；本教程不会自动分配新数据角色。

## 1. 先选对复现层次

| 层次 | 需要什么 | 能证明什么 |
|---|---|---|
| 论文图表重建 | Git 中的代码和 16 份轻量分析 JSON、Matplotlib | 表格数值、来源与图形一致 |
| 本地 producer 元数据校验 | 上述文件及本地 EqMotion producer manifest | 当前风险头对应 EqMotion，训练排除关系一致 |
| 检查点重放 | 本地完整缓存、检查点、环境及其 hash | 给定同一版本能重现分数、决策及指标 |
| 新训练复现 | 完整上游 producer 缓存、冻结配置、独立运行身份 | 新训练是否重现已登记实验 |
| 独立科学确认 | 新来源准入、校准与测试角色批准、完整上游排除关系 | 是否支持独立泛化或风险主张 |

前三项不等于后两项。第三方数据与大文件没有放进 GitHub；只有克隆仓库不能完成
全量模型重训。未观察到的 future label 不能按零误差填补。

## 2. 本机入口与线程

项目根目录为 `/Users/yangyue/Downloads/World`。使用该目录内的 arm64
`.venv-pytorch/bin/python`，不要使用默认 x86_64 Conda/Rosetta。

```bash
cd /Users/yangyue/Downloads/World
.venv-pytorch/bin/python -c 'import platform,sys; print(sys.executable,platform.machine()); assert platform.machine()=="arm64"'
```

正式训练入口在导入 Torch 前检查 macOS 架构。已完成训练使用计算线程 4、interop
线程 1、DataLoader workers 0。workers 0 不等于计算单线程。不要另开多个相同训练
进程挤占线程，也不要为了“检测设备”启动多进程资源探测。

已有实测证据：12 个新神经 cost heads、144,000 updates 和完整断点重放已完成。
这比 `import torch` 成功更强。本轮论文导出不重新宣称训练环境测试或 MPS 训练成功。
当前固定 CPU 实验的原地 `--resume` 不能随意切换设备、线程或配置；要换设备需新身份。

## 3. 最小可复现入口：不需要原始数据

以下命令从已有公开聚合报告重建，不读取原始轨迹或新增标签：

```bash
.venv-pytorch/bin/python scripts/build_m3w_evidence_manuscript_v2.py
.venv-pytorch/bin/python scripts/build_m3w_evidence_manuscript_v2.py --verify
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_evidence_manuscript.py tests/test_m3w_evidence_manuscript_v2.py
```

输出位于 `outputs/publication_readiness_2026_09/evidence_manuscript_v2/`。
`--verify` 逐字核对 JSON、CSV、Markdown 表格与 SVG，不重新抽 bootstrap。
Matplotlib 版本或字体变化可能改变 SVG；应诊断渲染环境，不能修改实验 hash 迎合结果。

本机额外校验预测器血缘：

```bash
.venv-pytorch/bin/python scripts/build_m3w_evidence_manuscript_v2.py --verify-local-lineage
.venv-pytorch/bin/python scripts/build_m3w_evidence_manuscript_v2.py --verify --verify-local-lineage
```

需要本地 `data/stage_cvpr2027_experiments/eqmotion_nested_v1/cost_views.json`。
核对 12 个 outer views 与 36 个 inner groups，不读取预测数组或 future targets。
缺文件应报错，不能生成一个空 manifest 让检查通过。

## 4. 有本地完整缓存时：已有实验重放

无需重训来找回已经完成的结果。先检查登记、身份与 manifest，再重放：

```bash
.venv-pytorch/bin/python scripts/run_m3w_fraction_square.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_fraction_square.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_fraction_square.py
.venv-pytorch/bin/python scripts/run_m3w_risk_ranking.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_risk_ranking.py
.venv-pytorch/bin/python scripts/audit_m3w_protected_joint_support.py --verify
```

这些命令需要此前已核验的私有缓存。原始版本的重放命令可能更新自身 heartbeat/
replay receipt，因此它不是纯只读文件操作；不要和同实验训练同时运行。本轮已复用
其完成证据，没有为了重写论文重复运行这些昂贵流程。

`--verify` 不应变成测试调参入口。出现 hash 不符，先查具体文件的变化；不能自动
刷新登记 hash。数值顺序、float32/float64 差异也要定位，而不是直接放宽容差。

## 5. 中断恢复与新训练

现有神经 head 训练入口支持以下实际参数：

```bash
.venv-pytorch/bin/python scripts/run_m3w_fraction_square.py --resume
.venv-pytorch/bin/python scripts/run_m3w_fraction_square.py --evaluate
```

`--resume` 恢复原身份、模型、优化器、采样器和随机数状态；已完成的实验不应重跑。
新试验先登记新配置、输出目录及上游 hash，保留旧结果。不要覆盖旧 checkpoint 来
制造“同实验改好了”的假象。该固定实验不提供任意 `--mps` 或 `--checkpoint-every`
命令行开关：间隔在冻结配置里，不能编造调用参数。

心跳和日志在 `data/stage_cvpr2027_experiments/<experiment>/` 下，具体以配置的
`output` 为准。检查 heartbeat PID 是否仍存在、事件 step 是否前进、CPU/RSS、
最近原子 checkpoint，再判断慢、等待还是停止。读取超时不代表进程死亡；没有确认
终止前不启动重复任务。中断后恢复最近已写入的 checkpoint，未写入部分可能重算。

运行完成要检查：进程退出状态、完整预算、有限 loss/梯度、抽样覆盖、未知标签
采样为零、配置与检查点一致、决策重放、每场景/种子的 easy 与缺失标签结果。
相关测试不是全套历史 tests；旧 suite 含会修改报告的非隔离测试，不能误称全部通过。

## 6. CREATE：先找已有资产，不重复提交

本轮本地 SSH 配置没有 CREATE 别名，M3W 远程项目目录与可用认证仍未核实。
这不表示 CREATE 没有结果或任务；本轮也没有声称读取了远程队列。
按 [CREATE 官方接入说明](https://docs.er.kcl.ac.uk/CREATE/access/) 使用已批准账号、
公钥及门户 MFA，并核对服务器指纹。不要把密码、私钥或令牌写入仓库。

访问恢复后，先只读查看本人队列、历史作业及已授权项目目录：

```bash
squeue --me
sacct -u "$USER" --starttime 2026-09-15 --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS
sinfo
```

先识别已有 config/checkpoint/log/split hash 和其科学角色，再决定复用还是新训练。
不能把其他课题目录当成 M3W。当前小 cost head 本地合理，不需要为了使用 HPC 而
搬运大缓存。大规模预测器、多 seed 或数据已在远程时，再按实际成本迁移。

正式任务经调度器提交，不在登录节点训练。依据
[CREATE 作业文档](https://docs.er.kcl.ac.uk/CREATE/running_jobs/) 和实际账号限制，
确认 partition、CPU/GPU、内存、时限、工作目录和日志目录；不在此编造可用队列。
以下只是待填写并核验的操作模板，本轮未提交：

```bash
sbatch --parsable <verified-m3w-job-script>
squeue -j <returned-job-id>
sacct -j <returned-job-id> --format=JobID,State,Elapsed,ExitCode,MaxRSS
```

作业脚本应使用远程原生 Linux 环境，不复制 macOS `.venv-pytorch`；固定计算线程与
申请 CPU 数一致、workers 0，记录包版本、Git SHA、输入 hash、checkpoint 与恢复命令。
调度器显示 COMPLETED 还不够，仍需核对实际产物、数值和科学 gates。

## 7. 当前最短科学路径

优先级一：确定独立来源准入及校准/最终确认角色，保留已暴露开发场景的身份。
优先级二：在批准后的独立场景检验 selected-harm 低估，不能直接用本轮结果改阈值。
优先级三：找到足够且过去可识别的交互支持，再测试 joint 方法；不对六个变化帧
反复调 pair weight。具体来源和角色问题已在之前提出，本稿没有代替用户决定。

目前没有新部署、独立风险保证或投稿达标结论。Stage5C、SMC 均关闭。
