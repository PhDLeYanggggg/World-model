# 候选预测尾部诊断：复现说明

本轮不改原评价规则，也不训练新的轨迹骨干。新训练的是八个正向损伤预测头，
与八个既有带符号风险头比较。已知未来只作为训练标签和诊断标签；未知未来不补成安全。
当前是 MovingDroneCrowd++ 的已暴露开发数据，观察8步、预测12步，图像坐标。
不能将其写成独立测试、旧SDD t50、秒级预测或物理安全证明。

## 环境与执行

在项目目录使用原生 arm64 环境；不要使用 x86_64 Conda。运行器在导入数值库前检查架构，
使用CPU4线程、inter-op1、DataLoader workers0；不启动资源探测或多进程加载。
森林实际设置为n_jobs=4，继承的启动日志forest_threads=1是旧字段，不是本轮拟合配置。

```bash
.venv-pytorch/bin/python scripts/run_m3w_moving_forecast_tail.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_forecast_tail.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_forecast_tail.py run
.venv-pytorch/bin/python scripts/run_m3w_moving_forecast_tail.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_forecast_tail.py
.venv-pytorch/bin/python scripts/run_m3w_moving_forecast_tail.py replay
```

设计、代码、协议和registration必须在真实拟合前提交。已有完成回执时，运行器拒绝覆盖。
首次新环境运行需要历史输入、冻结模型和对应哈希；仓库中的轻量报告不足以重建原始数据。
这不是无需数据即可从零运行的匿名包。当前HEAD保留登记文件，不能任意改旧版本来绕过校验。

若run中断，先看进程和私有heartbeat，不因终端超时启动第二份。确认原进程退出后：

```bash
.venv-pytorch/bin/python scripts/run_m3w_moving_forecast_tail.py run --resume
```

每个来源折保存一个原子checkpoint。恢复复用已完成来源折，尚未保存的单个森林需重拟合；
不声称支持森林内部树级恢复。verify会重新拟合所有新旧模型；replay只加载、预测和核对，
禁止拟合并检查公共文件哈希不变。完整历史测试套件没有在本轮重跑。

## 证据和边界

公共目录：`outputs/publication_readiness_2026_09/moving_forecast_tail_v1/`。
私有checkpoint、日志和预览图：`data/stage_cvpr2027_experiments/moving_forecast_tail_v1/`。
原始数据、缓存、模型checkpoint和原始行不提交GitHub。不要修改其他任务已暂存的文件。

这里的“捕获率”指在固定排序权重预算中覆盖多少正向预算超额成本，不是准确率、
轨迹改善、easy退化或安全通过率。并列分数均分权重，不能把它写成真实部署的选中子集。
oracle列使用结果标签排序，只是诊断上界，不是模型能力。没有正成本时记NA。

原2%门槛、样本支持要求和空效用对照NA全部保留；没有恒等回退认证。
没有打开新的C/H或保留测试，没有Stage5C或SMC。精确复现是工程证据，不是独立科学确认。
