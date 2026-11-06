# 前缀平衡预测器实验：本地运行与证据核查

## 本轮到底改什么

改的是实际 EqMotion K1 轨迹预测器的监督损失。输入、网络、随机种子、
来源隔离、抽样、优化器及4000步预算不变。不新增阈值，不对冻结轨迹做
时间渐变，不用测试终点作特征，也不是普通 residual/correction head。

完整12步可用标签平均误差占一半；前1/3/6步中有标签的前缀平均误差占
另一半。若前六步没有标签但后面有，保留完整可用误差的全部权重。
完全没有未来标签的行仍在既定抽样名单中，梯度为零；不将其记成零误差。

这是常规骨干训练修复，不声称前缀监督本身是新论文方法。

## 环境与运行

工作目录是项目根目录，解释器固定为原生 arm64 `.venv-pytorch/bin/python`。
CPU计算线程4，互操作线程1，数据加载工作进程0。不得用 x86_64 Conda/Rosetta
环境替代，不做 Torch 资源探测，不开启 DataLoader 多进程。

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_prefix_actor.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_prefix_actor.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_prefix_actor.py control --resume
.venv-pytorch/bin/python scripts/run_m3w_moving_prefix_actor.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_moving_prefix_actor.py forecast
.venv-pytorch/bin/python scripts/run_m3w_moving_prefix_actor.py evaluate
.venv-pytorch/bin/python scripts/supplement_m3w_moving_prefix_actor.py
.venv-pytorch/bin/python scripts/report_m3w_moving_prefix_actor.py
.venv-pytorch/bin/python scripts/run_m3w_moving_prefix_actor.py replay
```

先登记并提交定义，才能读真实训练数据。完成的 pilot 不重复运行；查验
pilot.json 的文件哈希与恢复记录即可。`--resume` 不改变预算和损失，不允许
从更好的中间模型代替第4000步。训练过程没有根据 donor 评分选检查点。

## 如何观察和恢复

本地私有目录：`data/stage_cvpr2027_experiments/moving_prefix_actor_v1/`。
`heartbeat.json` 记录当前 PID、模型、步数、loss、内存与耗时；`events.jsonl`
保留完整事件。每50步写心跳，每200步原子保存 checkpoint。检查点包含模型、
优化器、抽样随机状态、Torch随机状态和逐行抽样计数。中断后先核实原进程是否
终止，再对同一阶段使用 `--resume`，不要并行启动重复进程。

脚本有独占锁、5GiB磁盘保留、1GiB本轮私有产物上限和12小时阶段边界。
资源硬错误需要明确记录和恢复，不能把较短试跑称为完整实验。

## 如何核查结果

公开目录的 pilot.json 是真实100+100与200步恢复核对；control_refit.json
是完整4000步旧模型复现，不能用导入 Torch 成功或 loss 相近替代。旧日志
额外的20/40步仅是旧 pilot 停点，模型/优化器/随机状态等差异一律不允许忽略。

fits/ 应有六个完整4000步模型收据；forecast_freeze.json 应覆盖全部125540
行，并在 evaluate 前冻结。analysis.json 保留14个来源和所有录像的1/3/6/12
前缀结果及未知标签数。endpoint_diagnostic.json 只使用终点确有标签的行。

训练 CSV 的 legacy `mean_past_normalized_ADE` 名字来自旧实现，实际记录是
打包图像坐标下、未加权的可用标签 minibatch ADE。两种训练 objective 的
权重不同，不能直接用 loss 大小证明预测更好。主要比较是相同 CV 分母下的
逐来源完整可用12步 ADE 差，以及未知行轨迹分歧扣除后的保守下界。

重叠窗口不等于独立样本。来源 bootstrap 也只能是探索性区间：这些来源已经
暴露，交叉拟合训练角色存在重叠，不能据此称独立确认或三训练种子结果。

## 哪些结论仍然禁止

新预测器就算通过训练侧组件筛选，也不直接部署。必须重新形成因果成本标签、
拟合风险头并接受完整原规则检查，才有后续策略结论。原2%逐录像正伤害、easy
净损失及参考支持要求不变；空参考仍不可估，不允许恒等回退认证。旧失败保留。

本轮不读新的 own-B、C/H 或保留数据集，不使用测试终点构建目标。当前单位是
MovingDroneCrowd++ 原生观察8/预测12个标注帧、图像坐标；不是米、秒、true3D、
foundation、人工gold或物理安全证书。Stage5C execution 和 SMC 均关闭。

本地足以完成这个组件实验。CREATE 本轮只读观察超时，远端队列状态未知，
不因此推断作业失败，不提交、取消、重启或复制远端作业。
