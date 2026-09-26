# 本轮运行与恢复

## 研究范围

本轮检验训练内残差是否限制跨地点的代价修正。新训练的是风险估计头，
不是轨迹预测器；432 个两地点风险头、每个 2,000 步，共 864,000 步。
随后拟合 864 个固定的全局或上下文修正器。所有比较与阈值已预先注册。
训练支持检查通过不代表统计功效充足，更不代表方法取得提升。

沿用观察 8 步、预测 12 个标注步的开发协议，图像像素坐标。
不作秒、米、人工 gold、物理安全、true 3D 或 foundation 声明。
独立模型选择、风险校准和最终确认角色不开放，Stage5C、SMC 不执行。

## 环境与阶段

在项目根目录使用原生 arm64 `.venv-pytorch/bin/python`。
入口在导入 Torch 前拒绝 macOS x86_64；计算线程 4、interop 1、worker 0。
不得切换到旧 x86_64 Conda，也不得把 NumPy 路径当成本轮训练。

1. `--phase register` 固定方案；Git 提交后才能进行支持检查。
2. `--phase support` 检查全部 432 个拟合子集；提交支持报告后才能训练。
3. `--phase pilot` 完成首个风险头 100 步试跑，记录粗略计算成本。
4. `--phase train --resume` 从该检查点继续所有风险头，不重置种子或删掉困难切片。
5. 提交 `inner_prediction_freeze.json` 后运行 `--phase probes`。
6. 提交 `prediction_freeze.json` 后运行 `--phase evaluate`，然后 `--phase report`。
7. 执行结果绘图与完整复现检查。验证不会改变参数、阈值或部署。

上述阶段的入口均为 `scripts/run_m3w_european_nested_residual.py`。
不能在已运行阶段修改配置后继续冒充同一试验；更改须有新版本和原因记录。

## 中断恢复

私有目录为 `data/stage_cvpr2027_experiments/european_nested_residual_v1/`。
`heartbeat.json` 和 `events.jsonl` 记录 PID、时间、视图、内部留出地点及训练步数。
每 200 步和阶段末保存检查点，包括模型、优化器、随机状态、拟合预处理及抽样计数。
保留至少 10 GiB 可用磁盘。磁盘不足会明确报错并保留已有检查点；不要删除旧证据腾空间。
排除活跃进程后，可用以下命令恢复同一试验：

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_nested_residual.py --phase train --resume
```

独占文件锁会拒绝重复训练进程。运行慢不是卡死；先核对心跳、进程与检查点进展。
本轮已有 CREATE 只读队列记录，但未提交远程 M3W 训练，不得将队列可访问写成远程训练完成。

## 结果核查与安全同步

```bash
.venv-pytorch/bin/python scripts/plot_m3w_european_nested_residual.py
.venv-pytorch/bin/python scripts/verify_m3w_european_nested_residual.py
```

完整核查重放 432 个风险头、864 个修正器和 36 个评估组，独立核对 1,728 项均方误差。
绘图必须人工可视检查；程序重复生成同字节图只证明确定性，不证明排版可读。
测试数量、实际完成步数和结论以最终验证记录为准，本操作指南不是完成凭证。

Git 仅提交代码、配置、聚合报告、轻量指标和生成的 SVG。
不提交原始数据、逐行预测、检查点、缓存、预览 PNG 或虚拟环境。
保留仓库中无关的已暂存改动，按本轮明确文件清单提交。
