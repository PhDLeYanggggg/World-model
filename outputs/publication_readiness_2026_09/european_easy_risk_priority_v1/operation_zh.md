# 风险优先辅助训练：运行与恢复

本轮只改变辅助梯度上限。108 组、216 个头、每头 2,000 次更新、原有三个种子和数据角色保持不变。完整训练 **37576457** 已正常完成，第一组完整双臂重放 **37577264** 也正常完成。216 个检查点哈希、108 个对照模型状态、首组精确重放均核验通过。108 组因果动作已冻结，首次评价已完成，但完整重放和独立算术检查尚未完成，暂不称最终核验通过。

首次评价显示很小的同数量 ADE 正信号，但风险门槛未通过，部署不变。第一次动作重放在 28/108 组完全匹配后被 10 GiB 磁盘保护检查停止。空间恢复后已使用原命令从头重放，不重训、不删除旧数据、不降低保留线。最近 8 个文件的 47 项针对性测试通过。详情见 `readout_interim.md`。

重放提交曾在调度器回执阶段超时，之后从现有作业恢复了凭据，没有重复提交，也未修改其两小时时限。原失败记录保留。本机磁盘曾不足回传余量，11:47:03 UTC 复查已恢复到 14,236,880,896 字节，超过原来的 10 GiB + 800 MB 要求；未删除旧数据。每次实际回传前仍需复查。

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

1. 注册已在 `f22e98c8` 提交并推送，试跑已核验完成，不要再提交 `pilot`。
2. 试跑退出 0:0，两个检查点和配对采样已核验；主体 12.99 秒，调度总时长 49 秒，资源检查通过。
3. 完整训练 `37576457` 已完成，不要再次执行 `train`。
4. 第一组完整双臂重放 `37577264` 已完成，不要再次执行 `replay`。
5. 两次计算均已成功退出，检查点哈希已核验，轻量记录及训练报告已生成。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/manage_m3w_easy_risk_priority.py inspect
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/manage_m3w_easy_risk_priority.py train
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/manage_m3w_easy_risk_priority.py replay
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/manage_m3w_easy_risk_priority.py collect
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/report_m3w_easy_risk_priority_training.py
```

这些命令有先后条件，不能一次无条件全执行。提交器拒绝重复意图；查询超时不代表作业结束。完整训练配置为 CPU4、单进程、workers0、16 GiB、2 小时，试跑只申请十五分钟，不在登录节点做训练。每 500 次更新保存检查点与心跳，每组结束再记录进度。

## 训练之后

`decision_registration.json` 单独冻结了检查点恢复、因果动作与评价代码，不改变训练前已经注册的科学比较。恢复前必须有提交过的完整训练及重放记录；恢复仅接受正确路径和哈希的 216 个自有检查点，保留本机 10 GiB 余量。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/restore_m3w_easy_risk_priority_heads.py
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_easy_risk_priority_policy.py decide
```

动作清单已在 `b0d319cc` 提交后运行 `evaluate`。当前继续使用 `replay_decide` 和 `replay_evaluate` 重放；动作重放从头逐组验证，不能把中断前的 28 组称为全部通过。中断后的首次动作构建才使用 `decide --resume`。未来标签只供评价，不进入选择接口；底层加载器可将标签数组读入内存，因此不声称未来数据从未被加载。输入接口仅接收已登记的因果字段。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_easy_risk_priority_policy.py replay_decide
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_easy_risk_priority_policy.py replay_evaluate
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/verify_m3w_easy_risk_priority.py
```

每一步须确认上一项成功结束。完整验证要求两个重放记录都精确匹配；不要为生成报告绕过它们。若空间不足，保留完整文件，等待资源恢复或采用另行核验的计算路径。

## 中断恢复

先检查同一 job ID 的调度器状态、日志、心跳与检查点。如果仍运行或查询失败，不重启。若明确终止，保留既有检查点，核实原始注册、配置和输入哈希，再建立带新提交记录的恢复作业，使用同一计算脚本的 `--resume`。不覆盖旧提交凭据、不删除旧结果、不通过缩减组数来恢复。

运行入口为 `scripts/train_m3w_easy_risk_priority_portable.py`，恢复调用保留相同 `--home`、`--parent-home` 和 `--phase train`，只增加 `--resume`。必须通过调度器执行，入口会拒绝登录节点。当前未发生训练中断，也未提交恢复作业。

## 结果含义

训练报告中的 loss 是 fitting monitor。辅助梯度被限制、loss 下降、检查点重放一致，都不能替代 held-development 的同介入数量准确性与风险比较。后续必须先冻结因果动作，再读取新的 development 结果。独立 selection/calibration/confirmation 继续关闭。

项目仍为原始帧、图像局部坐标、检测器 silver 标注的研究系统。不能据此声称秒、米、人工 gold、物理安全、true 3D、foundation 或投稿完成。Stage5C 和 SMC 不执行。
