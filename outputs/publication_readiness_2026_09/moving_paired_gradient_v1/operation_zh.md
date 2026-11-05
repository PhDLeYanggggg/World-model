# 配对损失诊断复现

本实验已完成，不是新的训练结果，也没有更换部署模型。原2%规则保持不变，
空对照和支持不足仍不可估，不采用恒等回退豁免。

在项目根目录使用原生arm64环境：

```bash
.venv-pytorch/bin/python scripts/diagnose_m3w_paired_gradient.py --replay
.venv-pytorch/bin/python -m pytest tests/test_m3w_paired_gradient.py -q
```

复现读取现有私有缓存和九个已训练checkpoint，逐一检查数据绑定，再重算
完整训练分区上的梯度及十二次临时更新。不会保留更新后的模型，不会读取新的
测试结果，也不会覆盖公开报告。缺少私有数据时不能仅凭GitHub报告复现数值；
报告中的hash用于核对已有本地资产，不是数据下载地址。

CPU计算线程4、单进程加载、arm64检查和进程锁沿用已验证路径。日志和heartbeat
写入本地`data/stage_cvpr2027_experiments/moving_paired_gradient_v1/`，不提交GitHub。
当前诊断耗时约47秒，不是长时间训练。判断完成需看到进程退出成功、exact为true，
并核对analysis的hash，而不是只看heartbeat存在。

当前结论：两个分区有梯度冲突，但只有一个分区的投影更新同时满足误差改善、
决策代价不增加，未达到预先设定的2/3要求。不据此启动完整梯度投影训练。
