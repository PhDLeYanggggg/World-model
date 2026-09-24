# V2 原始轨迹审计与恢复

V1 完整审计因官方 60 份文件缺少文本类别列而中断，保留该失败和已完成的
4 份录像。V2 只增加这一明确格式支持，不推断缺失文本、不改类别编号或坐标。
输入仍使用 V1 下载并验证的同一份官方压缩包。V2 在独立目录从头重算，不把
V1 的局部成功包装成完整结果。

## 执行

```bash
.venv-pytorch/bin/python scripts/audit_m3w_european_squares_v2.py
.venv-pytorch/bin/python scripts/reconcile_m3w_european_squares_members.py
.venv-pytorch/bin/python scripts/audit_m3w_european_squares_v2.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_european_squares_arithmetic.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_european_squares_intake.py tests/test_m3w_european_squares_raw_v2.py tests/test_m3w_imptc_intake.py -q
```

若 V2 正常中断，用 `--resume` 恢复已经完成且身份一致的录像；若代码或数据
哈希改变，先诊断并另建版本，不覆盖旧结论。不要同时运行两个审计实例。
主程序逐录像保存私有结果。日志位置：
`data/stage_cvpr2027_experiments/european_squares_intake_v2/`。

`verification.json` 是完整原始录像重新解析和汇总一致性的证据。
`arithmetic_verification.json` 另外记录逐行哈希重放，以及每份录像固定选取
至多 3 个 agent 的独立集合计数。它不是全部轨迹的第二套计数：过去支持按
至多 64 个固定帧抽查，所选轨迹的未来标签计数全部核对，并直接核对速度。
样本选择不使用预测难度、误差或未来动作类别。

## 运行环境与边界

本轮是 arm64 Python 3.11.1、NumPy 2.4.6、pandas 3.0.3 的单进程数据审计，
不是 PyTorch 训练，也不是多种子模型验证。没有完整解压或巨大 episode 文件。
52 项针对性测试通过不等于整个历史测试集通过。

完整包校验、字段兼容、短轨迹保留、过去输入不随未来改变，是接入证据。
它们不等于场景独立性、真实时间、米制几何或在线检测因果来源已经验证。
原始文件、每轨迹哈希和缓存不进入 GitHub。公开文件只含代码、测试、汇总、
来源哈希和限制。没有模型提升、部署变更、Stage5C 或 SMC 执行。
