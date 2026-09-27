# 固定模型来源差距诊断：复现说明

## 这次实际做了什么

没有重训模型，也没有调整阈值。对上一轮冻结的108组新旧风险头，重新推理、
核对动作，再按训练来源决定的历史运动和特征支持分组重算误差。
未来标签是否完整、真实基线误差属于评估分析，绝不进入推理特征。

核心结果：风险不只来自特征范围外的输入，也不能仅用标签缺失解释。
完整标签仍贡献约61%的已观察伤害。高预测分歧、低转角切片风险偏高，
但这些是开发集诊断线索，不是可直接部署的新规则。

## 环境和运行

在项目根目录使用原生arm64环境，CPU4、interop1、DataLoader workers0。
不使用默认x86_64 Conda，不做资源探测，不启用多进程加载。
本机运行适合这次固定模型诊断；没有提交新的CREATE任务。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_floor_slices.py --phase run --resume
.venv-pytorch/bin/python scripts/report_m3w_fixed_floor_slices.py
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_floor_slices.py --phase replay
.venv-pytorch/bin/python scripts/verify_m3w_fixed_floor_slices.py
```

`--resume`复用已完成且身份核验一致的组；`--phase replay`重新计算并逐值比较，
不是把缓存读取冒充新训练。源文件或父证据哈希变化会停止，不覆盖旧证据。
每组保存聚合结果，心跳记录PID和完成数；保留至少10GiB磁盘空间。

## 证据位置

- `results.md`：全部切片、置信区间和逐来源结果。
- `conclusions.md`：证据支持什么、不支持什么，以及下一项可证伪假设。
- `summary.json`：固定来源名单、分母、计数和归一化误差总量。
- `diagnosis.json`：配对差值和伤害贡献分解。
- `verification.json`：针对性测试、独立重算和复现记录。
- 私有逐组结果位于`data/stage_cvpr2027_experiments/european_fixed_floor_slices_v1/`，不上传GitHub。

这不是独立测试，也不是新的部署升级；旧主检验失败不因分组方式改变而通过。
保持image-local、silver、obs8/pred12 rawstride12表述。Stage5C和SMC仍不执行。
