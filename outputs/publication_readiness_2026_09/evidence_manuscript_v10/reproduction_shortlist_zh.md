# 八项候选实验：本地复现与结果核查

这份说明复现 Moving A 已公开的开发实验，不打开独立数据。任务是观测 8 步、预测 12 步，均为标注步；距离沿用原图像坐标。三套预测器种子重复了同一批 34 个录像，共 102 个视图，不是 102 个独立录像。

## 运行前

在仓库根目录，用本机原生 Python 环境：

```sh
uname -m
.venv-pytorch/bin/python -c 'import platform, numpy; print(platform.machine(), numpy.__version__)'
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
```

Apple Silicon 上应显示 `arm64`，不用 Rosetta 下的 x86_64 Conda。这里运行的是 NumPy 选择器和评分，不会启动 PyTorch DataLoader。复训上游神经预测器时，仍需单进程加载（`num_workers=0`），并将计算线程和加载进程分别配置。

Git 里的实验注册、论文表和指标是轻量证据。真实计算还需要私有预测库、观测和目标标签、原来的切换决策及评分头；各脚本会检查父产物的哈希和行顺序。缺少这些文件时，不能把公开指标重读称为重新训练或独立验证。此前的私有文件不进入 Git。

## 复现本轮取证

当前提交的注册文件已固定，运行器会校验注册和源文件：

```sh
.venv-pytorch/bin/python scripts/run_m3w_shortlist_exclusions.py freeze
.venv-pytorch/bin/python scripts/run_m3w_shortlist_exclusions.py score
.venv-pytorch/bin/python scripts/run_m3w_shortlist_exclusions.py replay
```

`freeze` 只复用原评分头和预测器，为每个留出来源在六个校准来源上重选阈值。检查点写在私有 `data/stage_cvpr2027_experiments/moving_record_joint_actions_v1/causal_ledger/shortlist_exclusions/`；`heartbeat.json` 给出 PID 和最近完成的来源。中断后先检查 PID 是否还在运行，确认退出后重跑同一阶段即可复用已经冻结的来源检查点，不需要清空缓存。运行器也会在剩余磁盘低于 2 GiB 时停止。

`score` 生成轻量结果；`replay` 在同一冻结决策上重算。对比两份运行凭证的 `artifact.sha256`，以及分析文件本身的 SHA-256。数值复跑一致说明该开发结果可复现，不能代替独立数据确认。

## 构建英文附录

```sh
.venv-pytorch/bin/python scripts/build_m3w_shortlist_followup.py
```

这个命令只根据三组已核验的开发实验生成 Markdown、TeX 和来源绑定，不评分新数据。用本地 TeX 工具编译 `data/stage_cvpr2027_experiments/evidence_manuscript_v10/shortlist_followup_build/main.tex`，输出一页 `main.pdf`。上一节的六加二取证单独记录在实验报告中，不会混入三组附录数据。

## CREATE 什么时候需要

本轮校准和评分可在本机完成，没有提交 CREATE 作业。上游完整多种子训练如需用 CREATE，先确认输入数据和环境位于授权路径，并用 `squeue -j JOB_ID` 检查已有作业、`sacct -j JOB_ID` 核查结束状态。连接超时不等于失败，不要因此重复提交。训练通过已审计的对应作业脚本提交，保留作业号、配置、日志和检查点；恢复时用相同数据与配置，作业结束后再核查产物哈希和实际指标。已验证的远端汇总复算示例见 [CREATE 复算回执](../european_boundary_diagnostic_v1/create_recovery_20261001.md)，它不是本轮训练结果。
