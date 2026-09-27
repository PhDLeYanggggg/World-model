# 配对风险学习实验：本地运行与恢复

## 本轮做什么

保持预测器、保护基线、特征、来源划分和每个模型的训练预算不变，
只对比逐个体风险损失与当前群体汇总风险损失。108 组配对、216 个新
Torch 风险头，各训练 2,000 次更新。不是重新训练轨迹预测主干。

原始来源、历史缓存、检查点和逐行预测仅留本地。Git 中保留实现、
配置、哈希清单、轻量汇总和研究局限，不上传第三方数据。

## 环境与安全边界

在项目根目录使用 `.venv-pytorch/bin/python`。入口在导入 Torch 前拒绝
macOS x86_64/Rosetta；计算线程为 4，interop 为 1，数据 worker 为 0。
保留 10 GiB 可用磁盘；不要降低保留线或删除其他项目的数据来强行运行。

训练与决策通过单进程锁避免重复启动。每 500 次更新保存压缩的模型、
优化器、随机数状态和抽样计数。心跳及 PID 位于本轮私有目录
`data/stage_cvpr2027_experiments/european_query_excess_refit_v1/`。
读取 `heartbeat.json` 和 `events.jsonl` 判断实际进度；仅终端暂时没有
输出不能证明卡死。中断后使用同一配置和 `--resume`，不要重建注册。

## 分阶段命令

以下不是可直接从零顺序重跑的无人值守脚本。注册、训练冻结和决策冻结
之间有提交检查；已有产物不可覆盖，复现使用 replay 阶段。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_query_excess_refit.py --phase train --resume
.venv-pytorch/bin/python scripts/run_m3w_european_query_excess_refit.py --phase replay_fit
# 核对并单独提交 training_freeze.json 后才能生成决策
.venv-pytorch/bin/python scripts/run_m3w_european_query_excess_refit.py --phase decide --resume
# 核对并单独提交 decision_freeze.json 后才能读取评价标签
.venv-pytorch/bin/python scripts/run_m3w_european_query_excess_refit.py --phase evaluate
.venv-pytorch/bin/python scripts/run_m3w_european_query_excess_refit.py --phase replay
.venv-pytorch/bin/python scripts/run_m3w_european_query_excess_refit.py --phase replay_evaluate
.venv-pytorch/bin/python scripts/report_m3w_query_excess_refit.py
.venv-pytorch/bin/python scripts/verify_m3w_query_excess_refit.py
```

`replay_fit` 从头复现首对完整模型，不覆盖原模型；已存在复现检查点时
不要盲目重复启动。全部 216 个模型的预测通过 `replay` 重新提取。
正式冻结文件中的哈希必须与本地文件匹配；缺少私有数据时不得宣称完成
冷启动复现。完整旧测试目录包含会训练并改写历史报告的集成测试，本轮
只运行明确列出的隔离测试，不能写成完整旧测试集已经通过。

## 如何解读

这轮注册主指标存在读数前已披露的零介入分母问题，见
`primary_feasibility_addendum.md`。不得删除相关场景或把未定义写成零。
次要结果无论好坏均保留，不能替代主指标、校准证书或部署证据。

只有 12 个已开放开发地点。多组划分和重叠窗口不是独立样本。
统计以地点汇总后的配对 bootstrap 为基础。独立选择、校准和确认数据
仍关闭。数据为图像局部坐标、自动检测 silver，观察 8 步、预测 12 步，
raw-frame stride12；没有米、秒、人工 gold 或物理安全结论。
Stage5C 和 SMC 不执行。正式投稿仍由作者最终确认。
