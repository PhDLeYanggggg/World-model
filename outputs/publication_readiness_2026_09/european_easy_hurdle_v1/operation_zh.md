# 复现与恢复说明

本轮是两种 easy 风险监督目标的配对实验，不重训预测器。既有产物按 hash 复用；
训练源审计是本轮重新计算。未生成 summary 时，真实训练和开发评价都不得写成完成。

使用仓库根目录的 arm64 `.venv-pytorch/bin/python`。CPU4、interop1、workers0。
保留10GiB空闲，入口会拒绝越过这一底线；不要更改配置绕过。CREATE 需先确认独立
M3W目录、环境及允许的调度器资源，不借用 simulation 的目录或登录节点训练。

```sh
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_easy_hurdle.py --phase pilot
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_easy_hurdle.py --phase train --resume
```

训练前已完成登记与 fitting-only audit。pilot100步若已留下checkpoint，不重复无resume
启动；恢复时原命令加`--resume`。完整训练216头，每头2,000更新，不静默缩减。每500步
原子保存模型、优化器、随机状态和采样摘要，记录PID和心跳。中断保留已完成组。

全部训练完成后，先将training_freeze.json安全提交，再运行`--phase decide --resume`。
全部动作完成后，先提交decision_freeze.json，再运行`--phase evaluate`。不得交换顺序。
然后执行`--phase replay_fit`、`--phase replay_decide`、`--phase replay_evaluate`。
replay_fit只对第一组两头完整重训；其余组是推理重放，不写成全216头重训复现。

独立算术检查使用scripts/verify_m3w_easy_hurdle.py。冻结前生成报告使用
scripts/report_m3w_easy_hurdle.py。检查点、行级数据和原始调度器输出不进Git。
旧完整测试集有写报告/真实训练副作用；先运行本轮scoped tests，不宣称全仓库通过。
