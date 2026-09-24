# 零参考误差风险读出：运行和核验

这是既有风险森林上的 36 个新叶节点概率读出，不是重新训练神经轨迹预测器。
使用四个已经参与开发的 SDD 场景、三个种子、完整 175756 个历史可用窗口。
协议为观察 8 步、预测 12 步，stride12 原始标注步，坐标是 annotation pixels。
不能与历史 raw t50 混报，也不是独立确认或物理安全证明。

## 环境与恢复

在项目根目录使用原生 arm64 `.venv-pytorch/bin/python`。计算线程 4，
inter-op 1，单进程读取，无 DataLoader 多进程。真实 16-tree 试跑已完成，
随后恢复到全部 36 组，每组 128 棵树的叶节点读出。

```sh
.venv-pytorch/bin/python scripts/run_m3w_zero_atom.py --phase all --resume
```

私有产物位于 `data/stage_cvpr2027_experiments/zero_atom_v1/`。
`heartbeat.json` 和 `events.jsonl` 保存实际 PID、进度和时间；
`readouts/<site>_seed<seed>/<action>/checkpoint.joblib` 每 16 棵树保存一次。
进程锁禁止两个写入者。只有前一进程确实结束后才恢复，不能因运行慢另开重复任务。
恢复先检查所有绑定源码、配置、输入、上游模型和划分身份，不接受静默覆盖。

## 完成后的核验

```sh
.venv-pytorch/bin/python scripts/run_m3w_zero_atom.py --phase decide --verify
.venv-pytorch/bin/python scripts/run_m3w_zero_atom.py --phase evaluate --verify
.venv-pytorch/bin/python scripts/check_m3w_zero_atom.py
.venv-pytorch/bin/python scripts/report_m3w_zero_atom.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_zero_atom.py tests/test_m3w_cutoff_relative_risk.py tests/test_m3w_dimensionless_risk.py tests/test_m3w_net_easy_guard.py tests/test_m3w_native_metrics.py -q
```

重放检查完整选择数组、记录和聚合结果。另一个算术实现检查叶节点样本权重、
每个查询的配对干预数量、原单位风险约束及场景误差归约；它仍由同一执行者运行，
不是独立科研审查。定向测试不代表全部历史测试。

## 解释边界

零参考误差样本原本就在 easy 标签中，不存在已确认的标签遗漏修复。
新读出显式估计该事件，然后按既定零容忍规则否决有正估计概率的切换。
经验概率为零，只表示访问到的有限源叶节点没有这类加权事件，不是风险为零的保证。
所有原规则、守卫规则及同查询同数量对照均保留，不按结果挑选部署赢家。
未知未来和不完整标签不从索引删掉，也不能被算作安全。

DroneCrowd 保持独立确认关闭状态，IMPTC 仍隔离；不打开两者预测误差。
不执行 Stage5C，不启用 SMC。Git 仅提交代码、配置和轻量报告，不提交本地读出权重或数据。
