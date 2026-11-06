# 分角色成本模型与场景联合选择复现

本实验使用 MovingDroneCrowd++ 已暴露 own-B 的 8 个来源分量、24 段录像。
观察 8 步、请求预测 12 步；坐标仍为图像坐标。不是 SDD raw t50 复验，
不是米、秒、真实 3D 或 foundation 结果。原始 24 组主估计仍 NA。

需要此前已经校验的私有预测、特征和标签缓存。Git 只保存代码、定义和轻量
指标，不含第三方原始标注、缓存或权重；仅下载 Git 不等于已经复现数据。

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_role_separated_cost.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_role_separated_cost.py prepare
.venv-pytorch/bin/python scripts/run_m3w_moving_role_separated_cost.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_role_separated_cost.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_role_separated_cost.py verify
.venv-pytorch/bin/python scripts/run_m3w_role_separated_readout.py infer
.venv-pytorch/bin/python scripts/run_m3w_role_separated_readout.py evaluate
.venv-pytorch/bin/python scripts/report_m3w_moving_role_separated_cost.py
.venv-pytorch/bin/python scripts/run_m3w_role_separated_readout.py replay
```

入口拒绝 Rosetta/x86_64，固定 CPU 4 线程、interop 1、workers 0，不探测 MPS
资源，不创建 DataLoader 多进程。先执行真实数据 128+128 与连续 256 步恢复
对照，再训练两种同容量损失。两者初始化、来源/查询采样和 4,096 步预算相同。

`registration_v2.json` 是训练定义。v1 只做了与模型无关的数据准备；首次拟合
前已修正参考分支的动作不变性，原注册和数据哈希保留。v2 复用已验证的同一
数据，而不是把已缓存数组改称新转换。实际准备与断点试跑记录在 `executions/`。

训练和完整复训各 48 个模型、各 196,608 次参数更新。固定最终 checkpoint，
不看评价结果选 seed、模型或阈值。每 128 步保存，进度记录在私有目录的
`heartbeat.json`、`events.jsonl`。若中断，按原命令加 `--resume`：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_role_separated_cost.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_moving_role_separated_cost.py verify --resume
.venv-pytorch/bin/python scripts/run_m3w_role_separated_readout.py infer --resume
```

同一来源的全部录像从其模型训练、归一化和目标 RMS 中排除。输入仅为过去运动、
邻居和原有候选 rollout；未来有效性只用于训练监督或评价。条件成本与标签支持
分开学习，支持概率不是物理可见性或基线失败概率。

联合选择使用原有 6 个幅度，不新增 epsilon 动作。1 秒求解上限触发时回退到
已验证可行的独立策略并记录原因，不假称最优。每 250 个查询保存选择进度。
所有来源、种子和损失的选择先冻结，之后才读取评价标签。

`replay` 重新计算模型预测、验证缓存选择的可行性和同查询数量、重算指标，
不重新拟合，也不重新执行有时间上限的整数搜索。报告必须披露这个区别。
每次复现保存独立私有记录，不覆盖旧 PID/耗时。复现时不要同时生成公开报告。

每阶段最多 12 小时；磁盘保留 5 GiB，新实验私有产物上限 2 GiB。健康运行
不会因为慢而自动缩成 quick。安全评价保持原逐录像 2% all/easy 正伤害、
signed easy-net 和参考支持条件。空匹配效用仍不可估，不等于安全通过。
72 个录像视图是 24 段录像重复 3 个风险种子，不是 72 个独立场景。

主比较为两种损失的 joint-free 配对 completion-lower 差，先在来源内平均种子，
再对 8 个来源分量做 3,000 次 bootstrap。其他控制不能替代预先固定的主比较。
只有全部原风险视图和主比较都支持，才有后续候选证据；不是自动部署或投稿就绪。
不读取新 C/H、Moving20、DUT、DroneCrowd；不执行 Stage5C 或 SMC。
