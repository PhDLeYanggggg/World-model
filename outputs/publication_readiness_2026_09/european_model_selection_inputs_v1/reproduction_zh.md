# 独立场景输入复现

本次仅转换已冻结的六个模型选择场景。没有生成未来标签窗口，没有计算预测误差，
没有训练或选择模型。不能把这个步骤称为外部泛化成功。校准集、确认集与 DroneCrowd
均保持关闭。原始文件、输入数组和缓存不上传 GitHub。

在仓库根目录使用本机 arm64 环境：

```bash
.venv-pytorch/bin/python scripts/build_m3w_model_selection_inputs.py pilot
.venv-pytorch/bin/python scripts/build_m3w_model_selection_inputs.py build --resume
.venv-pytorch/bin/python scripts/build_m3w_model_selection_inputs.py verify
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_model_selection_inputs.py \
  tests/test_m3w_model_selection_input_evidence.py \
  tests/test_m3w_european_squares_source.py tests/test_m3w_european_squares_roles.py
```

实际完成的阶段不能覆盖。以上命令用于具备原始数据的尚未执行环境；已存在完成记录时
先检查对应 JSON，而不是删除后重跑。`build --resume` 会核验并复用最大的 pilot 录像。
若原进程仍健康，继续等待；只有确认原进程中断后，才对中断的 pilot/build 阶段使用
`--resume`。不要改变划分、schema、查询预算或注册文件。

私有目录是 `data/stage_cvpr2027_experiments/european_model_selection_inputs_v1/`。
`heartbeat.json` 记录当前 PID、阶段与已完成录像数；每个 `records/*/receipt.json`
绑定原始行、注册版本以及每个输入数组的哈希。公开 `verification.json` 必须同时满足
完整 28 个原始录像重新解析、逐数组一致及 `analysis_sha256` 与当前结果文件一致。
仅检查文件存在、缓存可载入或 toy tests 通过都不是这个复核的替代。

下一阶段仍须补齐当前模型所需的检测 confidence 历史与几何特征，核实训练来源和冻结
模型链路。禁止把新场景行 ID 伪装成旧 SOURCE 行，也不能用未来端点填缺失特征。
独立模型选择结果不是独立最终测试。坐标是图像局部坐标，时间是 raw-frame，不声明米或秒。
