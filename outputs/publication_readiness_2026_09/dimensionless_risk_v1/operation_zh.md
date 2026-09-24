# 无单位风险头对照：运行与恢复

这是新的 source-only 风险头训练，不是新 Transformer/EqMotion 训练，也不是
外部确认实验。72 个森林对应 4 个已设计暴露场景、3 个种子、3 个预测器、
2 个输入特征版本；每个128棵树。原有预测器和已冻结源数据抽样不变。

## 实验顺序

先登记完整矩阵与规则。运行第一组16棵树实际试跑，验证本地资源；随后
从 checkpoint 恢复到128棵树，完成全部72组。不是以试跑结果代替正式矩阵。

```bash
.venv-pytorch/bin/python scripts/run_m3w_dimensionless_risk.py --phase train --view coupa_seed17 --action damped_velocity_005 --arm native --stop-at 16
.venv-pytorch/bin/python scripts/run_m3w_dimensionless_risk.py --phase all --resume
```

已完成的训练记录和 checkpoint 哈希会验证后复用；未完成的森林从最近一次
16棵树 checkpoint 恢复。单进程数据载入，拟合使用4线程；不要使用 x86_64
Conda。并发启动第二个写入进程会被 runner.lock 拒绝。

## 查看真实状态

私有目录：`data/stage_cvpr2027_experiments/dimensionless_risk_v1/`。
其中 heartbeat.json/events.jsonl 保存 PID、训练组、树数量、六个目标的训练
MSE；trials 下保存 checkpoint 和完成凭据。仅有旧 heartbeat 不代表仍在运行，
必须核对 PID/任务会话。进程活跃且持续推进时不因训练慢停止。

预先存在的 .venv-pytorch、数据、源码和旧 checkpoint 已在本地。CREATE 当日
最近一次已保存连接记录为认证失败，因此远程当前任务状态未知。本实验本地
资源充足，未重复远程访问、提交作业或影响 simulation 项目的任务。

## 完成后的复核

```bash
.venv-pytorch/bin/python scripts/run_m3w_dimensionless_risk.py --phase decide --verify
.venv-pytorch/bin/python scripts/run_m3w_dimensionless_risk.py --phase evaluate --verify
.venv-pytorch/bin/python scripts/check_m3w_dimensionless_risk.py
.venv-pytorch/bin/python scripts/report_m3w_dimensionless_risk.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_dimensionless_risk.py tests/test_m3w_net_easy_guard.py tests/test_m3w_net_easy_risk.py tests/test_m3w_easy_moment.py tests/test_m3w_unit_free_prefix.py -q
```

decide 在读出新的 held-source 汇总结果前固定全部决策。训练来源和预处理不含
对应排除场景；但四个 SDD 场景早已用于研究设计，因此这仍是开发证据。
使用3000次物理场景配对 bootstrap，不把重叠窗口当作独立样本。

这个版本只改变风险头特征，未接入上一轮 canonical-prefix 新接口：后者改变
低速轨迹归一化，还存在 EqMotion 数值失败。把两者一起改会破坏单因素对照。
原生坐标下的 easy cutoff 也没有变成通用跨域安全标准。不能据此宣布完全
坐标不变、独立校准、外部泛化、部署、论文候选或真实3D成功。

源数据、checkpoint、训练/决策缓存留在私有目录；Git 只保留代码、配置和
轻量证据。DroneCrowd 保持关闭；Stage5C 和 SMC 保持关闭。
