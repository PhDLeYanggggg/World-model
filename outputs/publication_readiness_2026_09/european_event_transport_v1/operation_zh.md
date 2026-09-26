# 共同事件残差实验：运行和复核

本实验只改变残差的监督标签，不训练新的轨迹网络。使用已核验的
432 个内层风险网络和原外层风险模型，重新拟合 864 个固定岭回归修正器。
所有输入仍为过去信息；未来误差只作监督或评估标签。

## 本地运行

工作目录：`/Users/yangyue/Downloads/World`。使用原生 arm64 环境。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_event_transport.py --phase fit
.venv-pytorch/bin/python scripts/run_m3w_european_event_transport.py --phase evaluate
.venv-pytorch/bin/python scripts/run_m3w_european_event_transport.py --phase report
.venv-pytorch/bin/python scripts/plot_m3w_european_event_transport.py
.venv-pytorch/bin/python scripts/verify_m3w_european_event_transport.py
```

注意顺序：方案和注册锁必须先提交；拟合后，`prediction_freeze.json`
必须先提交，才能运行新结果读出。代码会检查该边界。不要通过删除锁或
改旧结果强行重跑。完整拟合结束前不能把单个场景称为完整实验。

`fit` 自动核验并复用已完成的逐视图收据，可在中断后再次执行同一命令。
双进程锁阻止重复执行。每个视图均更新私有 `heartbeat.json` 和
`events.jsonl`。保留 10GiB 磁盘余量。CPU4、interop1、workers0，不使用
数据加载多进程，不进行资源探测。文件位于：

`data/stage_cvpr2027_experiments/european_event_transport_v1/`

该目录中的逐行预测、缓存和模型系数不上传 GitHub。

## CREATE 与数据边界

本轮只进行授权范围内的 CREATE 队列只读检查，没有提交或修改远程作业。
固定低维拟合适合本地；这里的命令不是 CREATE 训练证据。
祖先模型的完整重放记录按哈希复用，不能写成本轮重新训练。

共同 easy 阈值来自外层三个训练场景。它可定义元学习器的监督标签，
但没有被用于重新训练内层 teacher，不能宣称每个共同标签都对内层留出行
独立。外层评估阈值保持原样，保留的独立选择、校准、确认数据未打开。

## 如何解读

`aggregate_metrics.json` 是本轮配对统计，`event_accounting.json` 检查
裁剪前标签差的线性投影恒等式。恒等式通过不代表提高预测，也不代表
证明根因。`verification.json` 只有实际完整重放后才生成。

主要统计按场景汇总三个种子后做 3000 次 bootstrap。六个角色划分相互
重叠，不能算作六个独立研究。点估计范围不是置信区间，没有显著负值也
不是安全性或等效性证明。轨迹、部署策略、Stage5C、SMC 均不改变。

当前是观察 8 步、预测 12 个标注步的 detector image-pixel 研究；不是
米、秒、人工 gold、物理安全、true3D 或 foundation 结果。
