# 因果运动特征对照训练：运行说明

## 实验边界

这轮重新训练的是108个风险头，不是重训整个轨迹预测器。只增加六个过去运动、
邻居上下文和预测分歧特征，其余训练设置、回退策略和2%风险预算保持不变。
不依据留出结果选阈值，不把未来标签完整度、真实误差或未来坐标用作输入。

## 运行环境

使用本机原生arm64的`.venv-pytorch/bin/python`，不要切到x86_64 Conda。
CPU计算线程4、interop1、DataLoader workers0，不进行资源探测或多进程加载。
每500次更新保存可恢复的完整状态，每组完成保存哈希绑定的轻量记录。
压缩是无损存储优化，不是降低模型精度、减少样本或缩短训练。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_causal_descriptor_refit.py --phase train --resume
.venv-pytorch/bin/python scripts/run_m3w_european_causal_descriptor_refit.py --phase replay
.venv-pytorch/bin/python scripts/replay_m3w_causal_descriptor_training.py --phase fit
.venv-pytorch/bin/python scripts/replay_m3w_causal_descriptor_training.py --phase causal
.venv-pytorch/bin/python scripts/run_m3w_european_causal_descriptor_refit.py --phase replay_evaluate
.venv-pytorch/bin/python scripts/report_m3w_causal_descriptor_refit.py
.venv-pytorch/bin/python scripts/diagnose_m3w_causal_descriptor_refit.py
.venv-pytorch/bin/python scripts/verify_m3w_causal_descriptor_refit.py
```

首次评价前，必须先把`decision_freeze.json`提交，才能运行`--phase evaluate`。
后续复现使用`--phase replay_evaluate`，不能根据已有留出结果重新挑选模型。
已完成组会校验后复用；中断组从最后一个原子checkpoint恢复，不从头重跑。
新状态压缩为`checkpoint.pt.gz`，必须用配套读取器恢复，不能误当普通未压缩文件。

## 查看结果

- `results.md`：固定主检验、全部对照、三种子和最差来源结果。
- `training_summary.json`：真实更新数、参数量、训练损失与未知标签抽样检查。
- `conclusions.md`：有效与无效的部分，不把训练完成当部署成功。
- `failure_analysis.md`：冻结读出后的配对误差与逐来源失败诊断，不改主检验。
- `operations.md`：本机真实运行耗时、内存、重放与存储记录。
- `verification.json`：重算、匹配、复现、针对性测试证据。
- 私有心跳和checkpoint位于`data/stage_cvpr2027_experiments/european_causal_descriptor_refit_v1/`。

保持至少10GiB剩余空间。若进程仍在更新日志和checkpoint，慢不等于卡死；
不得仅因观察超时重启。若真的遇到磁盘或运行错误，保留已完成组并诊断。
CREATE只按已有批准的连接和调度限制使用，不改认证环境，不在登录节点训练。

当前协议是image-local、silver、obs8/pred12 rawstride12，不能写成米或秒。
Stage5C和SMC不执行；独立校准与最终确认来源仍未开放。
