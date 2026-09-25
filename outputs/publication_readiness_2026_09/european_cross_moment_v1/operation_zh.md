# 交叉矩目标对照：运行与恢复

本实验只重训风险头，不重训轨迹预测器。先核验已有源数据、特征与前序模型
的哈希；缺少资产时明确报错，不把旧报告当作本轮计算。

## 环境

在项目根目录使用原生 arm64 `.venv-pytorch/bin/python`。入口在导入 Torch
前拒绝 macOS x86_64。CPU 计算线程 4、interop 1、DataLoader workers 0。
本轮规模适合本机，不改动 CREATE 上的任何任务。不要同时启动同模式训练。

## 首次运行

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_cross_moment.py --normalizer batch --prepare
.venv-pytorch/bin/python scripts/run_m3w_european_cross_moment.py --normalizer batch --audit --pilot
.venv-pytorch/bin/python scripts/run_m3w_european_cross_moment.py --normalizer batch --train --resume --replay --decide
.venv-pytorch/bin/python scripts/run_m3w_european_cross_moment.py --normalizer fitting --audit --pilot
.venv-pytorch/bin/python scripts/run_m3w_european_cross_moment.py --normalizer fitting --train --resume --replay --decide
.venv-pytorch/bin/python scripts/run_m3w_european_cross_moment.py --normalizer batch --verify
.venv-pytorch/bin/python scripts/run_m3w_european_cross_moment.py --normalizer fitting --verify
.venv-pytorch/bin/python scripts/report_m3w_european_cross_moment.py --normalizer batch --figures
.venv-pytorch/bin/python scripts/report_m3w_european_cross_moment.py --normalizer fitting --figures
.venv-pytorch/bin/python scripts/report_m3w_european_cross_moment.py --complete
```

两组模型及其决策都冻结后才允许读取新结果。第一组和前序 supported-pair
模型比较；第二组和第一组比较。不能按结果挑选获胜版本。

## 中断恢复

检查对应模式私有目录的 `heartbeat.json`、`events.jsonl`、PID 和已有
`heads/*/checkpoint.pt`。若原进程仍在推进，不重启。进程已停止时，重复
对应模式的 `--train --resume --replay --decide`，不要再次运行 pilot。
每 200 步原子保存模型、优化器、随机数状态和采样记录；恢复检查输入身份。
磁盘不足 10 GiB 或非有限 loss/gradient 时停止并保留最后有效 checkpoint。

## 证据边界

Checkpoint、逐行预测和完整分析保留本地；Git 只同步代码、配置、聚合指标
和报告。264 项 scoped checks 是实现检查，不是论文成功 gate；完整旧测试集
未在本轮运行。固定训练批次 loss 下降不等于排除场景的性能提升。
所有地点已属 development；这里不重新认证 Stage37 的历史部署结论。
结果只适用于 detector-track image-pixel 8/12 raw-stride 协议，不是秒、米、
人工 gold 或物理安全保证。独立确认集、Stage5C、SMC 不开放。
