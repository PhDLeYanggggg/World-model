# 冻结模型误差与训练集中度诊断

## 本轮做什么

只读取上一轮144个冻结模型，计算新模型相对原始/普通辅助模型的逐行平方误差
差，再按录像和轨迹汇总。变差量、改善量分别保留，不能先相互抵消后只报净值。
未来误差和easy只参与诊断标签；输入距离及其门限不接触未来标签。

对保存的最终训练批次计算录像分组梯度。分组损失仍除以完整批次大小，因此
各组梯度向量能加回总梯度。梯度范数质量不是因果影响；这也不是完整训练过程
或优化器更新的分析。径向输入范围只是粗略特征外推线索，不是可靠OOD证明。

## 执行和恢复

在仓库根目录使用原生arm64环境，CPU4线程、interop1、workers0。注册代码和
协议必须已提交，才能计算。真实首组试跑之后继续全量，不重复执行pilot。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_severity_transport.py --phase pilot
.venv-pytorch/bin/python scripts/run_m3w_european_severity_transport.py --phase run
.venv-pytorch/bin/python scripts/run_m3w_european_severity_transport.py --phase report
.venv-pytorch/bin/python scripts/plot_m3w_european_severity_transport.py
.venv-pytorch/bin/python scripts/verify_m3w_european_severity_transport.py
```

中断后用`--phase run`恢复。每组结果有不可改写回执；已完成组先校验再复用，
未完成组重新计算，不修改模型。心跳、运行日志和试跑记录在私有目录
`data/stage_cvpr2027_experiments/european_severity_transport_v1/`。
旧锁/心跳不能单独证明进程还活着；先查询实际PID，不因等待超时重启任务。

首组4个模型的组内时间15.5903秒，包含上下文载入及组内检查，不包含Python
启动、注册校验和之前的全部父级文件校验。`pilot.json`中
`includes_startup_and_hashing`仅指组内上下文启动/检查，不能解释为完整端到端
耗时。首组已被纳入最终36组，不是另一个更小的研究结果。

## 核查和解读

复核重新计算所有144个视图、误差质量守恒、432组梯度可加性及训练专属门限，
检查父模型文件不变，再执行相关测试并重画图表比较字节哈希。
独立选模、校准和确认数据继续关闭；旧版本验证明确标cached_verified。
结果仍是已暴露来源开发数据的诊断，不能恢复历史受污染的独立测试地位。

反复窗口、录像和角色均有依赖。质量集中度ESS不等于独立样本量；诊断标志
不等于统计显著、因果根因或部署成功。新训练步数是0，Stage5C/SMC仍关闭。
不上传原始数据、缓存、权重、视频图像或第三方素材；只同步代码和轻量汇总。
8步观察/12步预测仍为annotation steps，像素/检测标签不作米、秒、人工gold、
物理安全、true3D或foundation声明。正式论文提交仍由用户确认。
