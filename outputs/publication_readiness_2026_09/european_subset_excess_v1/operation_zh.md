# 本轮实验与复现说明

## 范围

本轮只训练风险决策头，不重训轨迹预测器，不替换部署模型。
数据是已经打开的12个开发地点；每个轮换使用4个预测器训练地点、4个
控制器地点、2个新头训练地点、2个开发留出地点。独立选择、校准和确认
数据保持关闭。历史已参与调参的结果不能重新称为独立测试。

观察8步、预测12步，步幅12个原始帧。图像局部坐标，检测器silver标签。
不声明米、秒、人工gold、物理安全、true3D或foundation。Stage5C和SMC不执行。

## 环境与中断恢复

使用仓库的原生arm64 `.venv-pytorch/bin/python`，计算线程4，interop1，
DataLoader workers0。入口在导入Torch前拒绝Rosetta/x86_64。每500次更新
保存压缩的模型、优化器、随机状态和抽样计数；每组写heartbeat。
保留10GiB磁盘余量。源码、配置、父实验或子集身份不一致时拒绝恢复。

运行位置为项目根目录。不要用默认Intel Conda，也不要修改已冻结文件。
本轮训练入口：

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_subset_excess.py --phase train --resume
```

只有原任务已经退出或被确认中断后才恢复；互斥锁会拒绝同实验重复运行。
日志、PID、heartbeat和断点位于
`data/stage_cvpr2027_experiments/european_subset_excess_v1/`，不提交Git。
本轮CREATE只做授权路径的只读队列检查，不提交远端训练。

## 顺序与核查

先注册并提交方案，再做preflight/pilot/train。训练身份提交后运行decide；
所有动作提交后才能evaluate。不得看留出结果后改权重、阈值或选checkpoint。
pilot的100次更新包含在每个头2000次总预算内，不是额外增加训练。

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_subset_excess.py --phase replay_fit
.venv-pytorch/bin/python scripts/run_m3w_european_subset_excess.py --phase replay
.venv-pytorch/bin/python scripts/run_m3w_european_subset_excess.py --phase replay_evaluate
.venv-pytorch/bin/python scripts/report_m3w_subset_excess.py
.venv-pytorch/bin/python scripts/verify_m3w_subset_excess.py
```

重放与原件做精确匹配；已有重放断点不可覆盖，请复用已核验结果。
全部训练从原始数据冷启动重建未在本轮重复执行，不能把缓存校验称为冷启动。

## 如何读结果

`results.md`记录所有预声明对照，不只展示最好的分支。
`locality_seed.md`给出地点和种子；`resources.json`记录实际计算成本。
`verification.json`是技术核查，不是研究假设或风险门槛通过证明。

风险排序分支在每个当前帧查询中切换同样数量，联合策略分支不一定同量。
原选中样本风险分母为零时仍为未定义；新增全体参考误差分母只用于诊断，
不得拿它替换原2%风险比例。训练损失下降不等于留出地点有效。
报告必须区分fresh_run、cached_verified和not_run。
本轮百分比不能与历史Stage37的t+50数字直接比较：数据、预测长度和比较
基座均不同，早期已暴露或存在污染的结果也没有被重新认证。
