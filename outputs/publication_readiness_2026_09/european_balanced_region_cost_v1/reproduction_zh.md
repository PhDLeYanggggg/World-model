# 完整区域训练的本地复现补充

本说明补充原有中文操作指南，不替换旧实验。此次实际完成54个成本头、
108,000次正式更新，首轮评价与独立进程重放均完成。科学门槛没有通过，
不能把命令跑完写成模型成功，也不能把旧结果或 oracle 写成新训练结果。

## 环境与资源

在仓库根目录使用原生 arm64 的 `.venv-pytorch/bin/python`，不要使用默认
x86_64 Conda/Rosetta。公共上游入口在导入 Torch 前检查 macOS 架构。
这轮实际使用单进程、CPU4、interop1，不启动 DataLoader 多进程，也不做
Torch 资源探测。不是 NumPy fallback，确实执行了 PyTorch 反向传播。

本轮正式训练454.86秒，峰值内存约10.21GB；额外重复访问训练区域行
12,275,011次。磁盘保留10GiB；单阶段健康运行上限12小时，不能因慢而
假装完成或缩小科学样本。CREATE 仅在资源或数据位置确有必要时使用；
新提交前先核对已有任务及迁移后的环境，不重复提交已有训练。

## 冻结资产与运行

真实重算需要私有 SOURCE 数据、缓存及旧冻结 checkpoint。公开统计文件
只能支持较窄的结果核查，不等于从原始数据重新训练。程序校验代码、
父实验、划分、预处理和权重哈希；不要改完文件后绕过检查继续旧实验。

新环境完整执行的阶段顺序如下，已经完成的阶段不要重复覆盖：

```bash
.venv-pytorch/bin/python scripts/run_m3w_balanced_region_cost.py register
.venv-pytorch/bin/python scripts/run_m3w_balanced_region_cost.py pilot
.venv-pytorch/bin/python scripts/run_m3w_balanced_region_cost.py neutral
.venv-pytorch/bin/python scripts/run_m3w_balanced_region_cost.py train
```

所有54个正式端点及训练冻结清单必须提交后，才允许第一次评价：

```bash
.venv-pytorch/bin/python scripts/run_m3w_balanced_region_cost.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_balanced_region_cost.py verify
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_balanced_region_cost_v1
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_balanced_region_cost_v1 --verify
.venv-pytorch/bin/python scripts/diagnose_m3w_balanced_region_shift.py
.venv-pytorch/bin/python scripts/diagnose_m3w_balanced_region_shift.py --verify
.venv-pytorch/bin/python scripts/report_m3w_balanced_region_cost.py --check
```

权重每100步保存，心跳每50步写入私有实验目录。先核实进程和最新心跳，
不要仅因观察超时就重启。确认进程终止且阶段未完成后，使用同一配置和
固定名单恢复；不要另建实验名偷偷重跑或更换阈值：

```bash
.venv-pytorch/bin/python scripts/run_m3w_balanced_region_cost.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_balanced_region_cost.py evaluate --resume
```

## 结果解读

本轮 mixed 已知标签 ADE 改善约0.9437%，但相对随机区域训练几乎不变。
相对物理候选的增量区间跨零，完整风险仅支持24/186个重复记录/头视图。
这不满足完整录像 all/easy 2% 要求，不能部署新头。空风险不是安全零。
三个种子是控制头种子，不是三条重新训练的完整预测流水线。

后续48组 oracle 检查只用未来标签作诊断，不训练、不输出推理特征。
其中32.40%和33.90%是理想逐行选择机会，不是模型成绩。它们的差值
也不能直接限制真实因果策略之间的差距。详见该诊断报告及数学反例。

当前坐标仍是图像像素、时间仍是 raw-frame。没有独立确认、metric/
seconds、true3D 或 foundation 证据。Stage5C 与 SMC 继续关闭。
不要提交原始数据、缓存、history/latent store、checkpoint 或第三方图像。
