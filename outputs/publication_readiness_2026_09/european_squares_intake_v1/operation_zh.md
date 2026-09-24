# European Squares 接入与恢复

更新：下载已完成。V1 完整审计在第 5 份录像因缺少文本类别列停止；下方保留
原命令作为历史记录。完整原始包改用 [V2 复现入口](../european_squares_intake_v2/operation_zh.md)，
不能把 V1 命令的中断视为完整成功。

本轮只做官方来源接入、原始轨迹结构和时间来源审计，不训练、不计算预测误差。
当前主协议仍是 SDD obs8/pred12、原始标注帧步长 12；这里额外检查的步长 1
和 12 仅用于来源可用性，不表示已经对齐真实时间。未分配数据角色。

## 环境

在仓库根目录使用 `.venv-pytorch/bin/python`。不使用 x86_64 Conda，
不启动 DataLoader 多进程，不运行发布方检测程序或模型权重。
本次表格审计用的 `pyxlsb==1.0.10` 安装到私有读取依赖目录，不改训练环境：

```bash
.venv-pytorch/bin/python -m pip install --disable-pip-version-check --no-deps --target data/stage_cvpr2027_experiments/european_squares_intake_v1/reader_dependencies pyxlsb==1.0.10
```

## 命令与顺序

```bash
.venv-pytorch/bin/python scripts/fetch_m3w_european_squares.py
.venv-pytorch/bin/python scripts/audit_m3w_european_squares_metadata.py
.venv-pytorch/bin/python scripts/fetch_m3w_european_squares.py --trajectories
.venv-pytorch/bin/python scripts/audit_m3w_european_squares.py --pilot
.venv-pytorch/bin/python scripts/audit_m3w_european_squares.py --resume
.venv-pytorch/bin/python scripts/audit_m3w_european_squares.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_european_squares_arithmetic.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_european_squares_intake.py tests/test_m3w_imptc_intake.py -q
```

命令列表是复现入口，不代表所有命令已完成。以 `analysis.json`、
`verification.json`、`arithmetic_verification.json` 及执行记录判断完成范围。
读取来源的每一步都检查固定版本和哈希。官方样例的 `--demo` 是独立检查，
不计入完整包规模，也不能替代完整审计。

## 中断与恢复

下载日志在 `external_data/EuropeanSquares_source_audit/`。`heartbeat.json`
记录 PID、已收字节和运行时间；`events.jsonl` 保留历史。下载中断后使用同一个
`--trajectories` 命令续传 `.part` 文件，不启动并行的重复下载。完整校验通过后
才生成最终 ZIP 和 `trajectory_manifest.json`。

结构审计日志在 `data/stage_cvpr2027_experiments/european_squares_intake_v1/`。
逐录像保留记录和代码身份。`--resume` 只接受同一身份的已有结果，读取缓存会
验证身份；`--verify` 会重新解析所有录像，不把读取旧报告当作重新计算。
若遇到半写入或身份不一致，保留错误文件并诊断，不能手工覆盖成成功。

每次只读取一个压缩包成员，不完整解压。磁盘不足、校验失败、重复 agent/frame
或未映射地点都是明确错误，不能用丢行、跳文件或猜测坐标来隐藏。正常慢任务可
继续等待；下载有 12 小时上限，磁盘保留至少 15 GB。

## 证据边界

元数据记录数不是有效轨迹数；窗口数不是独立样本数；相同城市的不同编号不自动
成为独立场景。官方代码显示逐帧检测流程，但不等于已验证所有录像的在线因果
处理记录。地理控制点存在也不等于米制准确。场景冲突、同源重复和时间映射未
解决前，不打开新来源的预测误差，不用于风险校准或确认集。

只提交接入代码、测试、哈希、汇总和报告。ZIP、CSV、表格、PDF、解析依赖、
逐轨迹索引和缓存留在忽略目录；本轮不改变部署，不执行 Stage5C 或 SMC。
