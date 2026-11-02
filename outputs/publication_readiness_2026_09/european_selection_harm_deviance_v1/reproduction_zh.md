# 伤害损失对照：运行与恢复记录

## 本轮实际完成

本轮不是只读取旧报告。新训练了18个Torch成本头，合计141,312次主要
优化更新；完整控制及首个生产者的新臂重训另计21,504次更新。实际磁盘
中断模型又独立完整重训8,192次，验证恢复后的模型、优化器、采样器、
访问记录、损失轨迹与随机状态和直接训练完全一致。

旧预测器、旧mean_absolute控制器和原始输入是cached_verified；18个新
成本头、推理和完整54组开发集评价是fresh_run。没有重新训练轨迹模型、
JEPA或Transformer，没有新增数据或搜索阈值。正式测试确认没有运行。

## 环境与命令

使用原生arm64的`.venv-pytorch/bin/python`，计算线程4、交互线程1、
DataLoader workers0。以下命令从项目根目录执行，依次对应恢复试跑、
完整训练、冻结推理、评价、独立复核和报告：

```sh
.venv-pytorch/bin/python scripts/run_m3w_selection_harm_deviance.py pilot
.venv-pytorch/bin/python scripts/run_m3w_selection_harm_deviance.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_selection_harm_deviance.py infer --resume
.venv-pytorch/bin/python scripts/run_m3w_selection_harm_deviance.py eval --resume
.venv-pytorch/bin/python scripts/run_m3w_selection_harm_deviance.py verify
.venv-pytorch/bin/python scripts/verify_m3w_harm_training_resource_resume.py
.venv-pytorch/bin/python scripts/report_m3w_selection_harm_deviance.py
.venv-pytorch/bin/python scripts/report_m3w_harm_policy_suppression.py
.venv-pytorch/bin/python scripts/plot_m3w_selection_harm_deviance.py
```

已完成的冻结阶段不可通过删除收据或覆盖检查点重新运行。恢复依赖配置、
输入、代码和状态的原有哈希。模型权重、训练数据和预测数组在被忽略的
`data/stage_cvpr2027_experiments/european_selection_harm_deviance_v1/`
中，不进入Git。上述真实训练还依赖其他私有SOURCE及旧模型资产；当前
仓库不是一条命令即可从公开原始数据冷启动的完整发布包。

## 实际中断与恢复

初始训练PID30865不是卡死：它在6个模型完成后触及既有10GiB磁盘储备
保护，在下一个模型2000/8192步保留完整检查点。没有删除数据、缓存或
系统文件，也没有改变训练预算或保护阈值。空闲空间自行回升后，由
PID32164恢复并完成。448.54秒只覆盖该完成进程，不包含先前中断进程。
初次日志和恢复日志都保留，空间波动原因尚未查明。

独立验证进程PID33193直接训练完整8192步，并与真正经历中断的最终
状态精确对比。CREATE队列也在本轮实际只读检查成功，但没有提交、
取消、修改远程任务；仅能证明队列可访问，不能据此声称远程GPU环境
已完成兼容性验证。本轮正式训练在本地完成。

## 结果应该怎样解读

双伤害损失相对原控制器的注册主要指标下降1.9774个百分点，95%区间
[-2.6565,-1.2590]，不是改进。已知标签ADE相对运动基座仍有0.08694%
均值增益，但不能覆盖主要对照失败。最差easy净退化降至1.6223%，
仍有134个all-risk和133个easy-risk视图违规，64个选中easy视图缺少
定义支持；只有30/252个重复录像视图满足完整原有要求。没有部署。

损失曲线来自实际训练，但不同损失不能直接按原始数值比较，曲线也不是
验证集指标。固定候选动作的SOURCE误差和不同选中集合的预测比值分开
报告；后者好转不等于同一样本上整体预测更准。

38个针对性单元与接口检查通过，完整仓库测试未运行。统计区间基于6个
已暴露开发场景的3000次重采样，不是独立最终确认。观测8步、预测12步，
采样间隔为12个原始帧，单位仍是图像像素。可见标签ADE不是完整12步ADE；
当前缺失标签界只覆盖整行未知，不覆盖部分未来内部缺失步。Stage5C和
SMC均关闭，不声明米、秒、人工gold、true3D、foundation或投稿准备完成。
