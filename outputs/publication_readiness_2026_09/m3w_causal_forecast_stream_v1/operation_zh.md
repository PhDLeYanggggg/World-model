# 本地流式读取与恢复检查

在项目根目录使用已有原生 arm64 `.venv-pytorch`。不使用 x86_64 Conda，
不启用 DataLoader 多进程；本轮 Torch 推理使用 CPU、4 个计算线程、1 个
inter-op 线程。先核对原训练权重、来源记录、过去输入缓存的校验值，不重复训练。

```sh
.venv-pytorch/bin/python scripts/check_m3w_causal_forecast_stream.py prefix
.venv-pytorch/bin/python scripts/check_m3w_causal_forecast_stream.py schedule
.venv-pytorch/bin/python scripts/check_m3w_causal_forecast_stream.py forecast --batches 3
.venv-pytorch/bin/python scripts/check_m3w_causal_forecast_stream.py forecast --batches 8
```

`prefix` 核对旧输入与新输入完全一致。`schedule` 扫描所有过去窗口并按录像
记录行数和校验值，不做模型推理。`forecast` 只对已经用于开发的 A 做真实
PyTorch 前向核验；前两次调用从三批恢复到八批，不触碰外部标签。已经完成的
批次直接复用，重新调用不会再产生首次写入的行数。各次运行收据在 `executions/`。

私有运行目录下的 `heartbeat.json` 记录当前录像或已完成权重。预测块先写临时
文件，重命名成功后才写完成标记，下一次按输入、权重和预测校验值续跑。只有
标记缺失的块才会重算；有标记但不匹配就停止检查，不能静默覆盖。不要同时启动
两个相同任务。当前命令仅用于本地核验，不是正式外部评价命令。

正式执行前先冻结独立校准和确认的角色、对照和风险规则。在 CREATE 先只读
检查任务和文件，核对数据/权重所在位置、调度器作业号、日志、检查点，避免
重复提交。当前连接查询超时，不能推断任何旧任务已退出。正式推理应按录像
流式保存或汇总，别保存全量特征。没有可靠远端资源时，按同一规则在本地分批
续跑，不因为慢减少种子或外部样本。所有指标仍用图像坐标和原始标注步。
