# 复现与恢复说明

本轮是两种 easy 风险监督目标的配对实验，不重训预测器。既有产物按 hash 复用；
训练源审计是本轮重新计算。未生成 summary 时，不得把训练完成写成开发评价完成。

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

## 当前 CREATE 路径

本地 pilot 因10GiB保留空间不足，在第一次真实更新前退出。不得删旧结果或降保留线。
独立 M3W CREATE CPU环境已通过真实优化器和精确恢复合成检查；不借 simulation 环境。
训练数据按组从内存上传，重启时逐一检查已有远端hash。连接中断不是训练失败。

```sh
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m scripts.export_m3w_easy_hurdle_create --resume
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m scripts.submit_m3w_easy_hurdle_create --phase pilot
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m scripts.inspect_m3w_easy_hurdle_create --phase pilot
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m scripts.submit_m3w_easy_hurdle_create --phase train
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m scripts.inspect_m3w_easy_hurdle_create --phase train
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m scripts.collect_m3w_easy_hurdle_training --phase train
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m scripts.submit_m3w_easy_hurdle_verification
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m scripts.collect_m3w_easy_hurdle_training --phase verify
```

这些是分阶段操作，不是可盲目重复的命令串。只有108组传输完成才提交pilot；只有真实
pilot完成且调度器exit0才提交train。已存在submission receipt或不确定提交时只检查，
不重复提交。训练使用已登记的同一fit实现、CPU4/interop1/workers0和原432,000更新预算。
远端训练完成不等于开发评价完成。模型与动作必须分别hash冻结并提交后才读取对应结果。
本地磁盘不足期间，远端模型和行级数组不批量回传。独立确认仍关闭。

本轮pilot前两次SSH在握手时失败，保留原receipt。远端只读确认无intent、receipt、脚本
或排队作业后，第3次实际提交成功；查看它需`--phase pilot --attempt 3`，不重新提交。
完整训练只提交一次。`create_training_freeze.json`记录远端216个checkpoint的hash；
核验job重新核对全部头，第一组两头从头重训4,000更新，并逐字段对比，耗时字段除外。
这不是全216头重复训练，也不是held预测重放；后两者的状态不得混写。
