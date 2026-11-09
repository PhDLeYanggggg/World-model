# M3W 当前论文与关键实验：中文复现入口

本文档对应 v124 研究稿。实验结果分三类：开发录像上的实测、使用未来标签的
事后上界、独立确认。前两类都不能写成第三类。DroneCrowd 的 15 个校准来源组
没有得到满足既定规则的策略，30 个独立确认来源组不用于以下命令。

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

参见 [v124 构建与校验](../outputs/publication_readiness_2026_09/evidence_manuscript_v124/README.md)。
主文、补充材料与生成图的命令均由对应的冻结读出生成。匿名复现只展示可公开的
汇总量，不包含原始轨迹、视频、模型权重或本机路径。当前工作稿没有提交。
