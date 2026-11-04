# 第二风险头否决实验：复现与解读

本轮没有训练新权重。用已经冻结的两个真实Torch代价头在原C训练来源重新推理，
由当前头提动作，另一个头只否决，不改动作或另选候选。2%门槛不变。
同数控制在每个查询中按原头的预测效用保留相同数量，不能仅匹配全数据总数。

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_cross_critic_veto.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_cross_critic_veto.py run
.venv-pytorch/bin/python scripts/run_m3w_moving_cross_critic_veto.py verify
.venv-pytorch/bin/python scripts/report_m3w_cross_critic_veto.py
```

上述计算已经完成，不要重复运行完成阶段。原进程已退出、阶段仍不完整时，
才可以使用原`run --resume`，保留已有选择缓存和逐任务凭据；不得删缓存伪装新实验。
每阶段有进程锁、PID/心跳、12小时界限、5GiB磁盘保留。当前规模适合本机，
不要因为CREATE连接超时就提交重复作业，也不要把连接失败当成远程作业失败。

读取`readout.md`、`interpretation.md`及`per_source.md`，不能只摘伤害下降：
减少伤害若丢失更多收益，效用仍然下降。主比较使用因果非空查询，完整策略比较
是另一个分母，二者不能随意相加。收益/伤害分解用完整策略的同一分母和配对区间。
空对照不可估，不是安全通过；348个录像风险视图都保留。

两个模型都见过C监督，因此这不是独立验证或校准。切换数匹配也不是幅度匹配；
动作幅度表仅作描述，不证明额外风险信息的独立贡献。本轮没有新的H测试阶段，
没有修改正式划分，没有新的模型部署。完整推理依赖本地授权数据、已有权重和缓存；
公开标量结果不是原始数据冷启动复现。像素和原生8/12标注步不是米或秒。
Stage5C和SMC都未启用。
