# 本轮运行与恢复说明

本轮固定原四场景训练的轨迹预测器和回退。A 场景组用于原预测器，B 用于新控制头监督，C 用于开发评估；六种方向都运行。这里只训练约2.5万参数的增益/伤害控制头，不重新训练轨迹网络，也不打开保留测试集。

## 环境与顺序

在 World 项目目录使用原生 arm64 `.venv-pytorch/bin/python`。CPU4、inter-op1、DataLoader workers0。入口通过已有运行库在导入 Torch 前检查架构，不使用默认 x86 Conda。

全新实验目录按以下顺序运行；已有完成产物时先用报告脚本核验，不覆盖 checkpoint。

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_producer_roles.py --phase prepare
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_producer_roles.py --phase pilot
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_producer_roles.py --phase train --resume
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_producer_roles.py --phase decide
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_producer_roles.py --phase evaluate --resume
.venv-pytorch/bin/python scripts/report_m3w_european_fixed_producer_roles.py
```

100步试跑计入每个头2000步总预算，不额外加步数。训练每200步保存 checkpoint 和心跳，目录锁阻止重复进程。中断后用相同配置和 `train --resume` 继续。不要删除已完成结果重来；文件哈希变化时先排查原因。

## 核查位置

私有目录为 `data/stage_cvpr2027_experiments/european_fixed_producer_roles_v1/`。
`heartbeat.json` 和 `events.jsonl` 记录 PID、进度、loss；`heads/` 保存新神经头，`ridge/` 保存线性对照，`decisions_complete.json` 是评估前冻结凭证。未来标签仅用于训练损失和评估，禁止加入推理特征。

完成训练不等于得到提升；报告必须核对144个头、72个 ridge 拟合、180组策略评估、checkpoint 回放和独立指标计算。loss 是训练量，不是验证集成绩。所有场景和负结果必须保留。

CREATE 本轮只查询队列，没有上传、提交或修改任务。授权的 M3W 远程产物目录仍未确定，不能据此断言远端没有产物；本地可运行，不因此中断。未来确需远程训练时沿用 simulation model 提供的限制，通过调度器运行，不在登录节点训练。

公开目录只包含代码、配置、报告和轻量汇总。数据、特征、checkpoint 与逐行预测不提交。公开汇总不能独立恢复私有原始数据。全部结论仍是图像像素、原始标注步长8/12的开发证据，不是米、秒、物理安全或独立确认。
