# 成本划分实验：运行记录

本轮是原生观察8步、预测12步的风险模型训练，不是旧SDD的t+50，也不是新的
Transformer/JEPA世界动力学训练。按用户决定保持原评价规则，不加入恒等回退认证。

## 已执行

协议与核心实现先提交为`e34a92ec62af2db5fa90b2b6c10d5d81492e8871`。旧8项MSE模型
按哈希核验复用；新训练16项分类划分模型，但最终输出仍是原权重下的风险均值，
不把分类概率直接当安全许可。

|阶段|PID|耗时秒|峰值内存字节|结果|
|---|---:|---:|---:|---|
|真实完整试跑及重复|40600|44.905|5961383936|正常退出、精确重复|
|16新模型+8旧对照|40759|71.694|6439043072|全部完成|
|全部24项重新拟合与核验|40897|74.573|6493437952|精确复现|
|已完成模型只读复现|41185|60.227|6467993600|禁止拟合，24项预测和指标精确一致|

只读复现保持当时41个公共文件完全不变。其后仅补充本运行记录与研究状态，
没有重新训练或更改科学结果。报告图已查看；v9已编译论文PDF哈希仍为
`d9babe3ddbac858bcc1ac30e3ba98b332bed625c78ff9231d1d42a3e30b11772`。
本轮`analysis.json`哈希为
`c19816ab24072260e27e0e3e270232c0d835f690e3576b6d5a42733f2a5c98bd`。

试跑单次完整64棵树模型拟合1.899秒。本机足够，没有新建CREATE作业；没有通过
降级规模缩短运行。固定模型预算全部跑完。31项针对性测试通过，4.36秒；未重跑
整个历史测试套件。首次合成测试的三个失败及修正已经单独保留，不隐去。

原生arm64 `.venv-pytorch`、Torch线程4、interop1、DataLoader workers0。保存的
森林参数`n_jobs=4`；继承通用日志的`forest_threads=1`是旧标签而非模型实际参数。
无resource probing。逐模型检查点、日志和恢复凭据位于忽略的私有目录。

## 命令

从项目根目录执行。已完成阶段禁止覆盖；当前复核使用最后的只读replay命令。
只有实际中断且缺少完成凭据的阶段可以用`--resume`。

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_cost_partition.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_cost_partition.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_cost_partition.py run
.venv-pytorch/bin/python scripts/run_m3w_moving_cost_partition.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_cost_partition.py
.venv-pytorch/bin/python scripts/replay_m3w_moving_cost_partition.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_cost_partition_probe.py tests/test_m3w_cost_partition_reporting.py tests/test_m3w_peer_path_probe.py tests/test_m3w_peer_path_reporting.py tests/test_m3w_candidate_neighbor_probe.py tests/test_m3w_candidate_neighbor_reporting.py
```

`registration.json`是初始预检查登记，没有实际拟合；有效训练登记为修正合成测试后
的`registration_v2.json`。两个版本及说明均保留。不能静默替换旧模型或实验配置。

## 结论边界

主模型内部筛选0/3。最大上下文风险预测误差降低35.80%，但决策代价仅改善0.284%，
且新增3个有害接纳，不能称可部署提升。原2%规则、空支持不可估及旧失败均保持。
没有新增保留集读数。精确复算不等于独立统计确认；论文附录不代表已重编译PDF。
