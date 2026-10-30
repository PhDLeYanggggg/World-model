# 冻结选择后的神经预测使用取证

本轮是主实验结果之后登记、取证数值计算之前冻结的诊断，不是追认主实验预注册。
它固定全部动作和模型，只问收益来自哪里、伤害估计错在哪里，不训练或调策略。

首次完整运行PID25357，用时101.95秒；另一个进程PID26550完整重放118.48秒。
72组旧独立策略取证、288项动作hash和全部汇总一致。PID仅是历史记录。
真实数据、源模型权重以及主实验的私有动作检查点需要本机合法存在。

只核验公开聚合包：

```bash
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_frame_allocation_attribution_v1 --verify
```

从仓库检出后，如需完整重放，先用 `--restore` 展开这份聚合JSON，再运行：

```bash
.venv-pytorch/bin/python scripts/run_m3w_frame_allocation_attribution.py verify
.venv-pytorch/bin/python scripts/report_m3w_frame_allocation_attribution.py
```

前一条实际重新推理并从冻结动作取证；后一条只生成报告和图，不重算科学结果。
新诊断没有完成凭据且确实中断时，使用 `run --resume`；不得覆盖已完成实验。
使用arm64 `.venv-pytorch`、CPU4、interop1、workers0。私有日志和heartbeat位于
`data/stage_cvpr2027_experiments/european_frame_allocation_attribution_v1/`。

解释结果时必须保留三个区别：

1. 删除神经动作到floor，只衡量固定策略中的实际神经使用，不等于重新优化纯物理策略。
2. 比较两种策略的神经使用变化时，共同使用的未知神经结果先抵消，再对剩余差异做界；
   不是用两个下界相减冒充一个新下界。
3. 单行风险超限不必然等于整个查询超限；未知和空查询不能作为已知安全样本。

当前真实结果支持拟合阶段的神经使用增益，但主策略仍有324个已知easy违规录像视图，
不部署，不称独立泛化或A会投稿候选。原始数据、权重、缓存和动作检查点不上传GitHub。
Stage5C未执行，SMC未启用；不能声称米、秒、人工gold、true3D或foundation。
