# 因果情境风险偏差实验

## 目的与边界

上一轮未支持普遍的录像梯度主导或径向分布外解释。本轮固定既有神经模型，
检查七类过去运动、近邻和冻结预测分歧特征能否解释可迁移的风险估计偏差。
每个基模型比较整体偏差修正和固定岭回归情境修正，不搜索阈值或正则系数。
共144个对齐视图、432个冻结估计器、864个新闭式拟合；新Torch优化步数为0。
这不是新神经网络训练，也不是用NumPy替代Torch后宣称神经训练成功。

未来风险只作拟合/评估标签，不进入特征。情境分位点只使用已知、预测分歧
为正的拟合行。无邻居明确编码缺失，不伪造距离。邻居最多8个，这是既有数据
结构的上限，不能解释为全场景完整密度。每个留出地点都从整个拟合链排除。

基模型在用于拟合偏差的行上已经训练过，因此属于in-sample residual probe，
不是独立校准。不能拿训练过外层留出地点的其他模型构造假OOF预测。若结果
为正，仍需后续单独注册嵌套OOF实验；当前不能调整部署或打开独立确认数据。

## 执行

仓库根目录使用arm64环境，CPU4线程、interop1、workers0：

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_context_residual.py --phase pilot
.venv-pytorch/bin/python scripts/run_m3w_european_context_residual.py --phase fit
```

注册文件必须先提交。pilot完成首个视图的6个闭式拟合，之后fit复用这些回执，
继续全部36组。模型系数及逐行预测保存在忽略的私有目录。所有预测完成后先
提交prediction_freeze.json，再执行：

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_context_residual.py --phase evaluate
.venv-pytorch/bin/python scripts/run_m3w_european_context_residual.py --phase report
```

中断时重新运行当前phase；已有不可变回执经哈希核验后复用。不要因观察超时
重启进程。私有目录保存heartbeat、events、完整系数和预测；至少保留10GiB。

## 验证

verify_fit重新拟合全部探针并逐项核对系数、训练标签/特征哈希与 held预测；
verify_eval重新计算全部36组指标。检查D/H/D_E逐项不变，H_E仍在[0,H]。
完整验证脚本串联这些步骤、1728项直接MSE核算和作用范围匹配的测试：

```bash
.venv-pytorch/bin/python scripts/plot_m3w_european_context_residual.py
.venv-pytorch/bin/python scripts/verify_m3w_european_context_residual.py
```

不能只跑语法检查就称实验复现成功。首个视图真实试跑约13.96秒，包含组内
数据载入和检查，但不含注册及完整父级预检；6个拟合计入864个总预算。

每个角色组合先在地点内平均3个seed对比值，再对4个地点重采样3000次。
这是已暴露来源开发证据，窗口、角色组合和情境分组有依赖，不能作为独立
确认或因果机制证明。负结果与不支持的分组保留，不挑最好角色。

CREATE本轮仅只读查询，未提交或改变作业。本地优先；若实际内存/耗时需要
再按授权调度，不因慢就伪称完成。原始数据、缓存、权重、逐行预测不上传。
仅像素/annotation steps；不声称米、秒、人工gold、物理安全、true3D或
foundation。Stage5C/SMC关闭，正式投稿仍由用户最后确认。

## 本轮读数

864个拟合与36组读数均已完成。注册提交b7a829eb，预测冻结提交3ac1ae56；
后者先于本轮留出读数。完整运行记录313.8133秒，不含父级预检且复用pilot，
不能当作整个实验的总耗时。没有新增Torch优化步，也没有改变轨迹输出。

完整输入的情境修正对原模型：6个MSE区间中2正、0负、4跨零；点估计范围
-1.8796%到+0.9351%。对整体偏差修正也是2正、0负、4跨零。未达到全部6个
组合均为正的预注册要求，不能部署。这里是风险估计MSE，不是轨迹FDE改善。
tail/coverage的“无负区间”检查通过，不等于证明安全或不劣；部分点估计仍负。

速度变化和近邻接近速度的部分偏差方向会重复，但分组/窗口彼此有依赖，
且幅度修正并未稳定有效。下一步先设计并审计嵌套OOF残差来源，不挑有利
特征、不用外层留出地点训练过的模型拼接假OOF，不打开独立最终测试角色。
