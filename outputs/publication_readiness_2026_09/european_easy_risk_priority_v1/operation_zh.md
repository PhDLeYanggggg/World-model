# 风险优先辅助训练：运行与恢复

本轮只改变辅助梯度上限。108 组、216 个头、每头 2,000 次更新、原有三个种子和数据角色保持不变。初始真实试跑作业为 **37575963**；最后核实状态是 `PENDING / Priority`，不能写成已经训练或卡死。

## 本地环境与检查

工作目录：`/Users/yangyue/Downloads/World`。使用本机原生 arm64 环境，避免默认 x86_64 Conda。以下 21 项针对性测试已通过，包含真实小型 Torch 更新、匹配对照和中断恢复，而不仅是导入测试：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m pytest \
  tests/test_m3w_easy_risk_priority.py \
  tests/test_m3w_easy_risk_priority_report.py \
  tests/test_m3w_easy_hurdle.py \
  tests/test_m3w_easy_hurdle_portable.py -q
```

这不是全部历史测试，也不是外部泛化结果。

## CREATE 执行顺序

数据已经位于 CREATE，无需再复制 4.91 GB 输入到本机。新实验目录独立于原实验及 simulation 项目。提交前读到个人配额 50,000,000,000 字节、已用 21,457,948,193 字节；不使用共享文件系统容量代替个人配额。

1. 注册已在 `f22e98c8` 提交并推送，试跑已经提交，不要再提交 `pilot`。
2. 观察同一作业，待 `COMPLETED / 0:0` 后检查真实试跑耗时、内存和配对样本记录。
3. 资源满足后只提交一次完整训练，继续试跑检查点，保留全部 108 组。
4. 完整训练完成后，提交一次第一组完整双臂重放。
5. 两次计算均终止成功、检查点哈希验证通过后，收集轻量记录并生成训练报告。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/manage_m3w_easy_risk_priority.py inspect
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/manage_m3w_easy_risk_priority.py train
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/manage_m3w_easy_risk_priority.py replay
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/manage_m3w_easy_risk_priority.py collect
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/report_m3w_easy_risk_priority_training.py
```

这些命令有先后条件，不能一次无条件全执行。提交器拒绝重复意图；查询超时不代表作业结束。作业配置为 CPU4、单进程、workers0、16 GiB、2 小时，不在登录节点做训练。每 500 次更新保存检查点与心跳，每组结束再记录进度。

## 中断恢复

先检查同一 job ID 的调度器状态、日志、心跳与检查点。如果仍运行或查询失败，不重启。若明确终止，保留既有检查点，核实原始注册、配置和输入哈希，再建立带新提交记录的恢复作业，使用同一计算脚本的 `--resume`。不覆盖旧提交凭据、不删除旧结果、不通过缩减组数来恢复。

运行入口为 `scripts/train_m3w_easy_risk_priority_portable.py`，恢复调用保留相同 `--home`、`--parent-home` 和 `--phase train`，只增加 `--resume`。必须通过调度器执行，入口会拒绝登录节点。当前未发生训练中断，也未提交恢复作业。

## 结果含义

训练报告中的 loss 是 fitting monitor。辅助梯度被限制、loss 下降、检查点重放一致，都不能替代 held-development 的同介入数量准确性与风险比较。后续必须先冻结因果动作，再读取新的 development 结果。独立 selection/calibration/confirmation 继续关闭。

项目仍为原始帧、图像局部坐标、检测器 silver 标注的研究系统。不能据此声称秒、米、人工 gold、物理安全、true 3D、foundation 或投稿完成。Stage5C 和 SMC 不执行。
