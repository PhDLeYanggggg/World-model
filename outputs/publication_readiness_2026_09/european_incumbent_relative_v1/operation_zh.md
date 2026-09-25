# 本轮实验与恢复说明

当前实验只使用已经开放的 EuropeanSquares 开发来源，比较“重新选择预测”与
“是否值得覆盖原策略”。没有开启独立确认集，不更新部署模型。

## 训练与恢复

使用项目中的原生 arm64 `.venv-pytorch`，CPU计算线程4、interop线程1、
DataLoader workers0。不要使用 x86_64 Conda/Rosetta 环境。

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_incumbent_relative.py --phase prepare
.venv-pytorch/bin/python scripts/run_m3w_european_incumbent_relative.py --phase pilot
.venv-pytorch/bin/python scripts/run_m3w_european_incumbent_relative.py --phase train --resume
.venv-pytorch/bin/python scripts/run_m3w_european_incumbent_relative.py --phase decide
.venv-pytorch/bin/python scripts/run_m3w_european_incumbent_relative.py --phase evaluate --resume
.venv-pytorch/bin/python scripts/report_m3w_european_incumbent_relative.py
```

已完成的实验只重跑报告器。不要在已完成的目录再次执行 pilot。
训练中断时保留目录，使用同一配置的 train --resume；不改配置、不删除 checkpoint。
已有完成回执会先校验；训练中的 head 从自己的 checkpoint 恢复。
独占锁防止同一目录同时运行两个阶段。心跳、PID、事件日志位于私有实验目录：
`data/stage_cvpr2027_experiments/european_incumbent_relative_v1/`。
报告器还需要该目录内已通过的 scoped_tests.log 和 scoped_test_files.json。

## 检查结果

先查看 completion_checks.json，再看 results.md、gates.md 和 conclusions.md。
配置和检查点通过不代表科研假设成立；所有负结果都必须保留。
每组置信区间使用4个来源地点的3000次配对bootstrap，不能把重叠窗口当独立样本。
add-only/remove-only是预先定义的诊断分解，不是看完结果后挑一个部署。

CREATE只做已授权的只读队列检查。本轮规模适合本地，不在登录节点训练，不上传
到未经核实的项目目录。Stage5C不执行，SMC不启用。
