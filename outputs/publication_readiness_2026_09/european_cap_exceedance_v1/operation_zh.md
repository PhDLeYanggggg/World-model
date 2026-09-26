# 因果超限事件实验操作说明

## 实验边界
这个实验预测“实际 easy harm 是否超过冻结模型预测的 all-harm”，不是
预测哪个轨迹一定正确，也不是风险保证。主协议仍为观察8步、预测12步，
原生标注步长、检测器图像像素。独立选择、校准和确认数据都不打开。

## 环境与恢复
在项目根目录使用原生 arm64 的 `.venv-pytorch/bin/python`。入口在导入
Torch 前阻止 Rosetta/x86_64；主进程设置4个计算线程、1个互操作线程，
不用多进程 DataLoader。检查点每200次更新保存一次，包括优化器、
随机数状态、预处理参数和已抽样行数。中断后使用 `--resume`，不能删除
检查点后把重新开始称为恢复。磁盘不足10GiB时保留已有产物并停止依赖工作。

私有目录为 `data/stage_cvpr2027_experiments/european_cap_exceedance_v1/`。
`heartbeat.json` 记录 PID、状态和步数，`events.jsonl` 保留历史。
心跳文件不是进程仍存活的证明，必要时同时查询该 PID。

## 固定执行顺序
以下命令是复现入口，不应在已有进程运行时启动第二份任务。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_cap_exceedance.py --phase support
.venv-pytorch/bin/python scripts/run_m3w_european_cap_exceedance.py --phase pilot
.venv-pytorch/bin/python scripts/run_m3w_european_cap_exceedance.py --phase train --resume
```

配置和原始注册记录已冻结，实施修正另存 `implementation_amendment.json`。
训练前必须提交支持检查；训练后先提交 `prediction_freeze.json`，再运行读出。
脚本主动检查这两道边界，禁止用结果反调本轮超参数。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_cap_exceedance.py --phase evaluate
.venv-pytorch/bin/python scripts/run_m3w_european_cap_exceedance.py --phase report
.venv-pytorch/bin/python scripts/plot_m3w_european_cap_exceedance.py
.venv-pytorch/bin/python scripts/verify_m3w_european_cap_exceedance.py
```

验证会重新构建输入、恢复全部检查点、逐值比对预测、重算开发场景指标和
报告，再运行限定测试。它不会重训模型，也不表示全仓库旧测试全部通过。
训练完成和研究假设获得支持必须分开看。

## 结果解释
六个 producer/controller 组合分别报告，三个种子在每个留出场景内平均，
再对四个场景做3000次配对 bootstrap。组合之间重叠，不能合成六个独立
成功案例；滑动窗口也不是独立样本。全部负结果、无类别支持和损失曲线
都要保留。这里的二分类概率不能直接替代 expected harm，更不能未经
独立校准就作为部署阈值。

## CREATE 与 Git
本轮先只读查询已有 CREATE 队列，不提交或重启远程任务。是否迁移训练
根据实际试跑成本决定。仅提交代码、配置、报告和轻量统计；原始数据、
特征、逐行预测和检查点留在私有目录。其他项目已暂存改动不得一起提交。
