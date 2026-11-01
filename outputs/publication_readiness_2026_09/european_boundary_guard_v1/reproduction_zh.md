# 两侧风险边界监督：复现与当前边界

本实验冻结当前 TDEF 排序器及所有预测路径，只改变 coherent critic 的训练损失。
两组从同一已完成 own-choice 权重开始，保留成本和 fixed-initial-choice query
回归；boundary 组额外学习原 2% 边界两侧。不是最佳 baseline 类别分类，不是
新轨迹训练，也不能把训练用 BCE 输出说成已校准的概率或安全保证。

## 本地运行

用原生 arm64 `.venv-pytorch/bin/python`，不要用 x86_64 Conda/Rosetta。
CPU4/inter-op1/workers0；入口先查架构，不做设备资源探测或多进程加载。
原始 SOURCE、完整因果特征、冻结预测、own-choice 和 TDEF 权重需在本地存在。
注册文件绑定定义和依赖；已完成阶段拒绝覆盖。

```sh
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_boundary_guard.py pilot
.venv-pytorch/bin/python scripts/run_m3w_boundary_guard.py train
.venv-pytorch/bin/python scripts/run_m3w_boundary_guard.py eval
.venv-pytorch/bin/python scripts/run_m3w_boundary_guard.py verify
```

只有实际中断后才对同一阶段加 `--resume`。不要因终端观察超时重复启动任务。
检查同一 PID、heartbeat 和输出增长；记录文件或锁文件本身不能证明仍在运行。
每50更新保存优化器、随机状态、完整采样器及损失轨迹，预算不能因慢缩减。
单阶段12小时边界可恢复，保留10GiB磁盘与1GiB私有检查点上限。

真实试跑已完成400次更新，包含两组50步中断后恢复至100步与各自直接100步。
恢复的科学状态、完整因果预测和动作精确相同。其资源估计不算效果评价。
正式预算为36个模型、208320次主更新及14336次完整重训验证，必须全部完成。

## 公开轻量复现

完整结果及数值复核产生后，运行：

```sh
.venv-pytorch/bin/python scripts/verify_m3w_boundary_public_report.py
.venv-pytorch/bin/python -m pytest \
  tests/test_m3w_boundary_guard.py \
  tests/test_m3w_boundary_guard_evidence.py \
  tests/test_m3w_boundary_public.py
```

报告从公开标量包重建，提前禁止 Python 的项目私有输入读取，仅允许写出结果图。
这不是操作系统隔离，也不是重训或独立科学复现。结果未齐时，相关检查明确 skip。
重新推断所有新成本和重算动作得分与公开报告重建是不同的验证步骤。

原149510 targets、74 recordings、12 localities 和72组不变。主要对照是完整
boundary-minus-moment completion-lower；不能用 raw 均值或同切换数量对照救回。
仍要求每个原录像的 all/easy 风险，空参考支持不可估，不能改成安全零值。
三个成本种子共享 forecaster43。所有结果是已暴露 SOURCE 的开发拟合结果，
不是独立测试、SDD t+50、米、秒、human gold、true3D 或 foundation 证据。

CREATE 只在实际资源或数据位置需要时使用，先核对已有任务，再通过调度器提交，
保留同一注册定义、job ID、配置、日志和恢复方案。本轮资源可本地承担，无需重复
提交远端任务。原数据、特征、缓存、权重及第三方资料均不上传 Git。
Stage5C 和 SMC 均不执行。
