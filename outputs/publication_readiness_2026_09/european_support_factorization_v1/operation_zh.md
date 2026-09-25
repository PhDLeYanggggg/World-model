# 支持条件拆分实验：运行与恢复

## 本轮究竟做什么

本轮复用已校验的预测器、234 个控制头及同一批开发场景。不是新训 Transformer，也没有重选阈值。只将原来的支持条件拆成：历史运动、模型预测分歧、分别满足两者、同一批来源同时满足两者。所有新策略都保留最新观察步的停止保护。

结果必须分清两个参照：比原来失败的联合过滤好，不等于比未加该过滤的 stop 策略好。报告同时给出这两个比较，以及同一录像同一当前帧、完全相同介入人数的风险排序和随机对照。

## 运行环境

从项目目录运行 `.venv-pytorch/bin/python`。使用原生 arm64 环境，不使用默认 x86_64 Conda。当前实验为 CPU 四线程、inter-op 一线程、DataLoader worker 为零。磁盘不足 10 GiB 时保留已有结果并报错，不删除原始数据来凑空间。

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_support_factorization.py --mode batch --phase decide
.venv-pytorch/bin/python scripts/run_m3w_european_support_factorization.py --mode fitting --phase decide
.venv-pytorch/bin/python scripts/run_m3w_european_support_factorization.py --mode batch --phase evaluate --resume
.venv-pytorch/bin/python scripts/run_m3w_european_support_factorization.py --mode fitting --phase evaluate --resume
```

前两个命令必须全部成功，才允许新结果读出。不要同时启动多个阶段：程序使用文件锁，避免重复进程。两个完整决策库一共 36 组，每组 26 个策略，共 936 个评估视图。其中 288 个是旧策略的重放锚点，不应冒充新模型。

## 中断恢复

私有路径为 `data/stage_cvpr2027_experiments/european_support_factorization_v1/`。查看 `heartbeat.json`、`events.jsonl` 中的 PID、阶段和最后完成的组，再检查进程是否仍在运行。只要进程仍正常产出就等待，不启动副本。

决策阶段重跑时核对已有数组，而不是覆盖成新策略。评估阶段加 `--resume`，只读取并验证已完成的组，再继续缺少的组。不要修改已注册的配置、核心代码、协议或测试后继续旧结果；这些内容都已进入身份哈希。每组结果为不可变 JSON，并由完成清单绑定。

## 报告校验

测试文件清单见 `completion_checks.json` 的 `scoped_test_files`，私有测试日志为 `scoped_tests.log`。这份清单覆盖本次改变及其依赖，不表示运行了整个历史测试目录。报告命令只有在全部结果和相关测试齐备时才成功：

```sh
.venv-pytorch/bin/python scripts/report_m3w_european_support_factorization.py
```

核查：936 份决策逐标量重放、288 个旧决策一致、144 份坐标误差数组独立计算、12,528 项指标独立归约、1,728 项旧指标一致，以及 4,608 项拒绝成本分解恒等式。完成清单还保存报告、测试和结果的哈希。

每个视图的置信区间使用 3,000 次来源场景 bootstrap，而不是把重叠轨迹窗口当成独立样本。三个既有训练种子保留，但这些视图相互相关，也不能当成 36 次独立验证。所有局部结果、最差场景和尾部误差都应保留。

## CREATE 与公开存储

本轮只通过已批准的只读路径检查 CREATE 队列。`simulation model` 未提供已知且获准读取的 M3W 远程产物目录，不能把“未检查”写成“服务器没有结果”。没有提交、取消、重启远程作业，没有借用其他项目数据或环境。本地资源足够，所以继续本地研究。

Git 只保存代码、配置、轻量汇总和方法图；不要提交源轨迹、预测缓存、决策数组、模型权重、第三方数据或虚拟环境。工作区中其他任务的暂存内容不属于本轮提交范围。

## 如何解释

这是已开放开发场景上的 image-pixel 观察 8 步、预测 12 步实验，原始标注帧步长 12。不是历史 raw t+50 的重新认证，不是秒、米、真实 3D 或 foundation 结果。标签来自已发布检测轨迹，不是人工 gold。保留独立模型选择、风险校准和最终确认角色关闭；不改变部署，不执行 Stage5C，不启用 SMC。
