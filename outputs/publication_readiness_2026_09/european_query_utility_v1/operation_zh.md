# 联合选择实验与结果核验

## 本轮做了什么

本轮不训练新预测器，也不重新挑阈值。使用已训练并核验的收益模型和风险头，
在同一录像、同一当前帧里保持切换人数不变，比较独立选择和收益感知的联合选择。
未来坐标、未来有效标签掩码不会进入决策函数。冻结全部决策后才核算结果。

注册提交：`0cffde4a`；决策冻结提交：`18801109`；首次结果提交：`ee1bca1d`。
这些提交保留了结果产生顺序，不是看到结果后重新定义成功条件。

## 复现顺序

工作目录为项目根目录，使用原生 arm64 环境。CPU 计算线程为4，interop为1，
DataLoader workers为0。不使用默认 x86_64 Conda，不探测 Torch 共享内存资源。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_query_utility.py --phase decide --resume
.venv-pytorch/bin/python scripts/run_m3w_european_query_utility.py --phase replay
.venv-pytorch/bin/python scripts/run_m3w_european_query_utility.py --phase replay_evaluate
.venv-pytorch/bin/python scripts/report_m3w_query_utility.py
.venv-pytorch/bin/python scripts/verify_m3w_query_utility.py
```

已有冻结结果的工作区使用重放，不覆盖或删除既有记录。真正首次执行需先注册、
提交注册文件，生成并提交全部决策冻结记录，然后使用 `--phase evaluate`。
程序检查这些依赖。每组保存小型动作掩码、完成记录和哈希，支持中断后续跑。
正式目标仍未完成；完成程序、单元测试或此开发实验不等于投稿准备完成。

## 实际资源与恢复

- 第一组试跑：5.94秒，最大RSS4,138,631,168字节，包含在后续108组内。
- 完整决策生成：PID37752，370.14秒，最大RSS11,163,058,176字节。
- 第一次结果核算：PID38418，42.08秒，最大RSS7,767,687,168字节。
- 完整决策重放：PID38606，372.33秒，最大RSS10,852,843,520字节。
- 完整指标重放：PID39214，42.52秒，最大RSS7,850,754,048字节，与首次结果一致。
- 私有压缩动作掩码：11,323,780字节；不上传到GitHub。
- 求解器22次未通过检查，保留原决策，不改变人数或删除这些样本。

第一次重放在5.42秒触发10GiB磁盘余量保护，未改动冻结结果。这是资源拦截，
不是进程卡死。仅删除本项目虚拟环境下可自动重建的 `__pycache__/*.pyc`，
清理前16,715个文件合计359,048,645字节。没有删除模型、包源码、数据、
checkpoint、环境配置或其他项目文件，没有降低磁盘保护阈值。恢复余量后
重新运行完整重放并核验全部108组一致。后续实验仍需检查磁盘余量。

本轮CREATE只沿用已授权配置进行了只读队列检查，返回码0；没有提交新作业、
更改认证、环境或在登录节点训练。该检查回执记录在前一步收益/伤害分解实验，
不能当作远程训练结果。

## 怎么理解结果

同等切换人数下，联合策略相对独立策略的ADE改善0.2322%，开发场景bootstrap
区间为[0.1225%,0.3586%]。这是小幅、有方向一致性的开发结果，不是独立测试结论。
实际风险超预算视图从82增至99，另有10个风险比例无定义，因此不部署新策略。
easy净误差保持不等于正伤害风险受到控制。未知标签样本也不能算作零误差。

检查 `results.md`、`conclusions.md`、`failure_analysis.md`、`verification.json`。
验证文件记录重放、逐画面约束、来源级统计重算和范围明确的测试。旧全量测试与
raw data冷启动重建若未执行，继续标为not_run，不以局部测试代替。

独立选择、校准、确认数据仍关闭。坐标是图像局部坐标，标注是detector silver，
观察8步、预测12步，raw-frame stride12；不是米、秒或human gold。
不执行Stage5C，不启用SMC，不把本轮结果表述为true3D或foundation成功。
