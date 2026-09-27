# 本轮复现与操作说明

本轮做的是固定强阻尼回退模型的增量价值诊断，以及216个线性成本预测头的真实拟合。
不是重新训练Transformer或JEPA，也没有启用Stage5C、SMC或新部署。

## 结果在哪里

- `results.md`：全部预注册对照、置信区间、逐种子结果和安全失败。
- `conclusions.md`：为什么神经预测有额外价值，但筛选后的风险仍不可靠。
- `conditional_moment_diagnosis.md`：事后核查预测伤害与实际伤害，不改变任何规则。
- `verification.json`：文件绑定、训练重放、无未来字段推理重放、统计复算和测试范围。

所有数值都是已经用于开发的12个来源场景上的结果，不是独立最终测试。
ADE使用图像局部坐标，观察8步、预测12步、raw-frame stride12；不能换称秒、米或历史t+50。

## 本地环境与复现

在项目根目录使用原生arm64的`.venv-pytorch/bin/python`。
训练入口会在导入Torch前阻止macOS的x86_64路径。计算线程4、interop线程1，
没有DataLoader多进程、没有Torch资源探测。下列命令不会重新选阈值：

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_floor_probe.py --phase replay_fit
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_floor_probe.py --phase replay_predict
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_floor_probe.py --phase replay_evaluate
.venv-pytorch/bin/python scripts/diagnose_m3w_fixed_floor_probe.py
.venv-pytorch/bin/python scripts/report_m3w_fixed_floor_probe.py
.venv-pytorch/bin/python scripts/verify_m3w_fixed_floor_probe.py
```

第一条重新拟合首个完整训练组并精确核对；第二条核对全部108组推理，并从真实输入字典
中移除未来标签、标签mask和未来误差；第三条复算全部统计。复核不等于冷启动重建原始数据。
需要本机已有且通过hash校验的祖先缓存和模型，公开仓库不包含这些大文件。

## 中断与恢复

每个闭式拟合完成后保存`ridge.pt`、预测和完成记录，持续更新heartbeat。
这不是按epoch运行的神经训练，不能把216个线性头写成216个新神经模型。
本轮原始拟合已经完成。只有原始运行中断时使用：

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_floor_probe.py --phase fit --resume
```

已完成组只检查hash后跳过，未完成组重新拟合。禁止删除或覆盖已封存的祖先结果。
目录中的`events.jsonl`记录PID和时间；每个运行阶段结束后核查终态，而不是只看进度消息。
磁盘必须保留至少10GiB；不足时先报告真实资源阻塞，不删除无关文件。

## CREATE与GitHub

本轮任务可以本地完成，无需GPU任务。CREATE仅沿用simulation model项目已授权交接做
有界只读队列检查，没有改认证、环境、权限或作业，也没有在登录节点跑训练。
检查回执存本地私有目录，不能把队列可读写成远程训练资产已经验证。

GitHub只保存代码、配置、轻量结果、图表和报告；训练缓存、逐行预测和模型留在忽略目录。
使用明确文件清单提交，不混入原有的大量无关暂存文件。报告中的正收益不等于研究目标完成：
当前预测风险筛选仍超预算，独立确认未运行，部署不变。
