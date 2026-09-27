# 训练内风险偏移拟合

本轮对216个已经冻结的神经风险头，各拟合两个非负偏移量。
这是解析参数拟合，不是重新训练神经网络；没有改变任何已有策略动作。
只用了原来两个拟合场景的标签，没有拿 held 场景选偏移或阈值。

复现使用项目根目录的 arm64 Python：

```sh
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m pytest -q tests/test_m3w_signed_bias_probe.py tests/test_m3w_signed_bias_verification.py
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/probe_m3w_signed_bias.py --resume
env PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/probe_m3w_signed_bias.py --replay
```

首次拟合前已经提交注册文件。现有注册不允许修改源模型、损失权重或数据角色。
`--resume` 检查并保留已完成的拟合，`--replay` 重新计算并逐组核对完全一致。
私有拟合参数、分组哈希、PID和heartbeat在同名data子目录，不上传Git。

结果和验证记录分别见 `results.md`、`verification.json`。训练目标函数还与
真实的原Torch损失作了对照，并用自动求导检查解析梯度。训练损失下降来自
这个解析最小化本身，不能解释成泛化提升百分数。

下一轮需要冻结“加偏移后的动作”，再与同样切换数量的旧分数策略比较，
才能判断它是否真的减少伤害。当前这一步尚未运行，部署没有升级。
独立校准和确认数据没有打开。仍是image-local、raw-frame、detector-silver
开发证据；不是米、秒、人工gold或安全保证。Stage5C/SMC都未启用。
