# DUT 冻结模型评估操作说明

本轮是已有神经预测器与成本头的真实推理和评分，不是重新训练。
预注册提交为 `ded982f5`。读取全部27个获准录像的过去可用窗口，
保留两个实际地点的分组，不把录像、窗口或三组随机种子当成新的独立地点。

## 运行与恢复

在项目根目录使用原生 arm64 环境：

```bash
.venv-pytorch/bin/python scripts/run_m3w_dut_readout.py --run
.venv-pytorch/bin/python scripts/report_m3w_dut_readout.py
.venv-pytorch/bin/python scripts/run_m3w_dut_readout.py --verify
.venv-pytorch/bin/python scripts/report_m3w_dut_readout.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_dut_population.py --check-results
.venv-pytorch/bin/python scripts/summarize_m3w_dut_evidence.py
```

只有上一运行进程确实结束后，才再次执行第一条。运行器使用文件锁防止重叠，
每128个查询保存原子检查点，完成的录像不会重复推理；中断录像从其最后检查点继续。
没有 `--resume` 参数，重复 `--run` 即检查身份并恢复。文件内容、配置、模型或
依赖哈希变化会拒绝恢复，不能绕过后把新结果说成原来的实验。

心跳在本地忽略目录 `data/stage_cvpr2027_experiments/dut_frozen_readout_v1/heartbeat.json`，
记录 PID、累计时间、录像和查询进度。密集录像计算一个块的时间可能较长，
应结合进程状态和计算活动判断，不凭心跳文件暂未变化就判断卡死或重启。
线程配置固定为计算4、互操作1、加载进程0，不调用硬件资源探测。

`--verify` 在新进程中重放每个录像第一、中间、最后的固定128查询块，
逐一对照输入、预测、决策和标签哈希。它不是第二次全量评估，也不是独立研究者验证。
报告脚本的 `--verify` 单独验证汇总算术和报告与冻结记录完全一致。
原始样本核对脚本不使用模型或窗口索引，直接读取哈希绑定的原始 CSV，
检查每个录像、每个固定模型的过去合格目标和未来标签覆盖数量。
最后一条只汇总所有固定模型，重复运行会拒绝不同的已有结果，不用于选最好种子。

## 如何解释结果

- 全部12个模型/成本头/种子组合都保留，不选择最好结果用于后续确认。
- 查询和目标由当前及过去决定，标签只能在全部模型完成决策之后读取。
- 原始标注属于离线标注历史，不代表已经证明传感器时间上的在线因果性。
- 不完整未来仍保留在推理分母中；完整路径指标有条件性，缺失部分另外给出相对收益界限。
- 本轮预注册的完整路径 ADE 与旧 SDD 可用点 ADE 不应混为一个主要结果。
- 只含两个地点的 bootstrap 仅为描述性重采样，不是稳健泛化或2%风险保证。
- 不重新定义简单样本，不根据 DUT 表现调整阈值，不读取 DroneCrowd 确认结果。
- 像素或数据集本地坐标不是米，标注步不是秒，Stage5C 和 SMC 保持关闭。

公开仓库只保存代码、配置、哈希及汇总。原始数据、录像、逐行缓存、模型权重和
私人连接信息不随本轮提交。CREATE 的独立 simulation 项目不受此本地评估影响。
