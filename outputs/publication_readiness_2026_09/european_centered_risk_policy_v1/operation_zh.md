# 风险偏移策略实验：复现与恢复

## 这轮实际做了什么

本轮不重新训练预测器或风险网络。读取上一轮已封存的训练内偏移量，
重新计算108组因果选择，并与每个查询中介入数量相同的未偏移策略比较。
只有选择结果全部封存并提交后，才读取相应开发集误差。
训练内损失降低不等于此次策略改善，必须以结果表和风险检查分别判断。

观察8步、预测12步，每步间隔12个原始帧。图像局部坐标和自动检测银标签，
不具有已验证的秒、米、人工金标或物理安全含义。
12个地点均属于此前已打开的开发数据，独立选择、校准和确认数据未打开。

## 环境

项目目录：`/Users/yangyue/Downloads/World`。
使用原生arm64的`.venv-pytorch/bin/python`，不能使用Intel Conda/Rosetta。
入口在导入Torch前检查架构。CPU线程4，interop线程1，DataLoader worker0。
已有模型检查点、偏移拟合和数据须在本地存在且通过哈希校验。
Git不包含这些大文件；仅从公开代码仓库克隆不能独立重建原始数据。

## 已完成版本的复验

```sh
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/replay_m3w_centered_risk_policy.py --phase replay
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/replay_m3w_centered_risk_policy.py --phase replay_evaluate
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/verify_m3w_centered_risk_policy.py
```

首次复验会保存独立运行回执。已封存的运行时间/PID回执不是可覆盖的输出；
若同一路径已有回执，应核验已有字节，不要为了重新计时改写封存文件。
代码、动作或统计结果不一致会报错，不能跳过比较后称为复现成功。
以后如需额外冷启动复验，应建立单独回执目录，保留本轮证据。

首次直接复验在第一组报错：动作数组一致，但身份记录里的元组与JSON列表
直接比较失败。复验适配器只将身份容器规范化为JSON形式，不改变数组、
模型、阈值或原始注册代码。该故障和修复回执保留，不能把首次失败隐去。

## 中断恢复

动作生成阶段使用`--phase decide --resume`。每组动作原子保存；恢复时先校验
已完成组的身份、父模型、训练偏移和数组哈希，不重复写入有效组。
运行锁阻止同一实验重复启动。先查看私有目录中的`heartbeat.json`、
`events.jsonl`和PID是否仍存活，不因等待或缓慢而重启健康进程。

私有目录：`data/stage_cvpr2027_experiments/european_centered_risk_policy_v1/`。
动作、详细行级统计和CREATE只读访问回执均在此目录，不提交Git。
公开目录保存配置引用、动作清单、汇总指标、运行回执、报告和核验封印。

## 首次实验顺序与边界

1. `--phase register`，安全提交并推送代码和注册协议。
2. `--phase pilot`，检查真实一组推理速度、内存、磁盘投影。
3. `--phase decide --resume`，完成全部108组，提交动作冻结清单。
4. `--phase evaluate`，第一次读取这些策略的开发集误差。
5. 完整动作复验、评价复验、独立数量/风险/误差/Bootstrap核验。
6. 更新README和研究状态，仅提交代码、配置、报告、轻量汇总。

保留10GiB磁盘余量。空间不足时保留有效产物并明确报告资源阻塞，不能删除
无关数据、降低预留标准或缩小样本后声称原实验完成。
本轮适合本机推理，CREATE仅只读查询，不提交重复训练任务。
Stage5C和SMC保持关闭。所有例行核验由项目执行流程负责，不需要用户逐项审计；
正式投稿仍由作者最终确认。
