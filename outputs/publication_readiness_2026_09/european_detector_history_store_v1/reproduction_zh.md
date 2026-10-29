# 过去检测历史缓存复现

使用仓库根目录的原生 arm64 `.venv-pytorch/bin/python`。不使用旧 x86_64
Conda，不启用 DataLoader 多进程。需要原始 EuropeanSquares 已审计 ZIP、
原始 SOURCE packed 输入和 fitting 录像清单；GitHub 不包含第三方原始数据。

```sh
.venv-pytorch/bin/python scripts/build_m3w_detector_history_store.py build --resume
.venv-pytorch/bin/python scripts/build_m3w_detector_history_store.py verify
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_detector_history_features.py tests/test_m3w_detector_history_builder.py
```

运行前先查看脚本帮助核对已有产物的不可覆盖要求。完整提取和单独重新解析的
数值报告分别是 complete.json 和 verification.json。原始读取、断点和心跳保存在
本地 data/stage_cvpr2027_experiments/european_detector_history_store_v1/。
禁止把整个私有目录加入 Git。完整提取 PID39447、独立重算 PID39736 均已退出。

初次试跑因局部变量遮蔽函数而在读原始数据前失败，记录仍保留。registration_v2
固定修复后的实现。新研究模型仍须各自在自己的 fitting 子集上估计输入标准化；
不能把本缓存的全局描述性统计当作跨划分训练统计。

这里只验证 supplied-export 的过去输入和精确对齐，不声称传感器 as-of、无误差
人工标注、模型性能或部署安全。预测目标、2% 完整策略风险阈值与封闭角色未改变。
