# 完整风险监督模型复现

这次改模型训练，不改风险评价：把旧训练池遗漏的未知标签样本加入监督，
使用原有轨迹分歧上界，不能把这些数值解释为真实未来误差。
网络、因果 120 维输入、零阈值、4,096 次更新、三个风险种子和来源隔离保持不变。
已知候选的特征与标签先逐值验证一致，再训练。

在项目根目录使用原生 arm64 环境：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_complete_risk_donors.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_complete_risk_donors.py prepare
.venv-pytorch/bin/python scripts/run_m3w_moving_complete_risk_donors.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_complete_risk_donors.py verify
.venv-pytorch/bin/python scripts/run_m3w_moving_complete_risk_donors.py infer
.venv-pytorch/bin/python scripts/run_m3w_moving_complete_risk_donors.py evaluate
.venv-pytorch/bin/python scripts/report_m3w_moving_complete_risk_donors.py
.venv-pytorch/bin/python scripts/replay_m3w_complete_risk_donors.py
```

定义文件必须先提交，入口会检查对应 Git 版本与注册哈希。这里只打开已授权的
producer-A 训练捐赠缓存和已暴露 own-B；不打开新的 C/H、Moving20、DUT 或
DroneCrowd。准备过程复用旧已知缓存，仅额外保存小型未知样本补充文件。

训练和完整复训各 24 个模型，各 98,304 次更新。固定最终 checkpoint，不在
测试结果上选模型或阈值。CPU 4 线程、interop 1、workers 0，无多进程加载。
每 128 次更新保存，记录 loss、梯度范数、访问数量、PID、内存和进度。
中断后原命令添加 `--resume`，例如：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_complete_risk_donors.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_moving_complete_risk_donors.py verify --resume
```

已完成模型会验证后恢复到相同步数，不重新选择种子。原始定义、训练数据和权重
不能在恢复中变动。最多运行 12 小时，磁盘保留 5 GiB；保护触发时保留原 checkpoint。

先完成全部训练和逐模型完整复训，再冻结 62,466 行完整策略，最后读取评价标签。
未知状态只用于训练监督和分组报告，不能作为推理过滤条件。
原始逐录像 2% all/easy 正伤害、easy-net、分母和支持要求不变。
空支持仍 NA；原始 24 组主估计仍 NA；旧失败保持失败。

可重复的 replay 入口只重新推理和重算指标，明确禁止拟合；必须在报告生成结束后串行运行。
每次使用独立私有执行记录，保留之前的 PID 和耗时。原注册入口的阶段执行记录
是单次不可变记录，不能用重复写入旧 PID 的方式伪装重跑；该入口限制保留并披露。
缓存、权重、原始轨迹和图保存在私有目录，不上传 Git。公开 JSON/Markdown 是
exposed-development 证据，不是独立测试；Stage5C/SMC 不执行。
