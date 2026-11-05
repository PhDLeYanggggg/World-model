# 组合决策损失：运行与恢复

本轮保留原规则，只修改神经风险头的训练目标。它不重训轨迹预测器，
不是新的 JEPA 预训练，也不执行 Stage5C 或 SMC。主实验是同一架构、同一
数据、同一起点和同一采样下的原损失与组合损失对照，不是阈值搜索。

## 环境

项目目录：`/Users/yangyue/Downloads/World`。
使用原生 arm64 `.venv-pytorch/bin/python`。入口在数值库导入前拒绝
macOS x86_64；本次采用 CPU4、inter-op1、workers0，不调用资源探测。
此前真实数据 pilot 已验证连续训练与中断恢复数值完全一致。

磁盘保留至少 5 GiB；本轮私有产物上限 512 MiB。不要删除旧数据、
覆盖历史结果或取消别的任务来腾空间。单进程文件锁防止重复训练。

## 命令

以下是新工作副本的顺序，不应在已完成目录反复运行训练命令：

```sh
cd /Users/yangyue/Downloads/World
.venv-pytorch/bin/python scripts/run_m3w_moving_composed_risk.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_composed_risk.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_composed_risk.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_composed_risk.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_composed_risk.py verify
.venv-pytorch/bin/python scripts/diagnose_m3w_composed_risk.py
.venv-pytorch/bin/python scripts/diagnose_m3w_composed_risk.py --verify
.venv-pytorch/bin/python scripts/report_m3w_moving_composed_risk.py
```

注册文件和指定代码须已提交且字节一致，才能正式训练。已完成的阶段
禁止覆盖。`evaluate` 正常退出也可能只是记录 `not_run`，不能据此说
留出场景已评估。只有预先规定的三个训练上下文全部通过拟合检查，才
进入已经暴露过的 nested-C 开发评价；该检查不是科学安全门槛。

## 中断恢复

先看进程与心跳，不要启动第二份：

```sh
pgrep -fl run_m3w_moving_composed_risk
cat data/stage_cvpr2027_experiments/moving_composed_risk_v1/heartbeat.json
tail -n 5 data/stage_cvpr2027_experiments/moving_composed_risk_v1/events.jsonl
```

只有原进程已退出、相应阶段尚未完成、配置与注册绑定未变时，才恢复：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_composed_risk.py train --resume
```

每 100 个完整优化器更新保存 checkpoint 和心跳。恢复模型、优化器、
采样器与随机状态；不保存一半执行的更新。健康训练可以继续等待。
不能把终端观察超时当成进程死亡，也不能为提速偷偷缩小训练预算。

## 结果与边界

模型与缓存位于 `data/stage_cvpr2027_experiments/moving_composed_risk_v1/`，
不进入 Git。公开目录保存轻量配置、哈希、训练曲线数值和报告。
`result_summary.json` 与 `README.md` 由已验证记录生成；完整状态同时
记录在项目根目录 `README_RESULTS.md` 和 `research_state.json`。

`fresh_run` 是本轮真实训练或计算；`cached_verified` 是旧资产经绑定
校验；`not_run` 必须保留原因。精确重放不等于独立数据验证。

原 2% all/easy 伤害、easy-net 与支持要求全部不变。缺支持和空对照
仍不可估，旧失败仍失败；不新增恒等回退认证，不打开保留 H 数据。
当前任务是 native obs8/pred12 图像坐标，不是旧 raw-frame t50，也不
支持米制、秒级、true 3D、foundation 或 human gold 声明。

统计可信度、独立支持、公开强方法对照和三个训练种子仍须分别补齐；
本轮单一 producer seed 的模型诊断不能替代这些论文证据。
