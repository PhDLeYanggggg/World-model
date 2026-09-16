# M3W 本地与 CREATE 操作记录

日期：2026-09-16。用途：当前可复现的工程步骤，不是正式预测实验教程的完成版。

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

这里的 8 观测/12 预测是可用性检查视图，尚未成为经批准的主实验协议。raw-frame t25 没有精确窗口，不能用邻近帧补成“t25”。所有视图可能重叠，不是独立统计样本。

## 正式实验入口与拒绝行为

```bash
.venv-pytorch/bin/python scripts/check_m3w_experiment_contract.py --report-dir outputs/publication_readiness_2026_09/experiment_contract/unapproved_rejection
.venv-pytorch/bin/python -m pytest tests/test_m3w_experiment_contract.py -q
```

当前第一条应返回 exit 2，因为协议仍未批准；这是安全拒绝，不是运行时卡死。第二条是 25 项合成/临时目录测试，不代表真实实验获批。不要通过把草案 `status` 改成 approved 或清空历史暴露来绕过：源数据、主评价规则、独立 calibration/confirmation 和用户决策仍须核实。

后续新训练代码需实际通过 `ExperimentContract` 打开对应用途的 recording；OOF gain/harm 使用的所有父模型也必须通过整 fold 来源检查。校准和最终测试前分别登记固定候选集合及 hash，恢复时身份必须相同；完成后不再作为 fresh confirmation 重跑。当前旧训练器尚未全部接入，不能宣称全仓库防泄露已经完成。详见 [实现与限制](experiment_contract/implementation_and_limits.md)。

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
