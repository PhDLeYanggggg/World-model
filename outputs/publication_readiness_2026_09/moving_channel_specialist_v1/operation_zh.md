# 本轮模型修复与复现

## 这次究竟改了什么

保持原评价规则。没有加入“与基线相同就认证安全”的豁免，没有提高 2% 门槛，
也没有把缺少 easy 参考样本的结果从 NA 改成通过。

先重算已冻结模型的决策边界，再把全体风险和 easy 风险拆成两个独立回归头。
两个头各 32 棵树，总数与原共享 64 棵树相同。另训练共享 32 棵树的控制组。
仍然预测原 signed risk，不用未来终点、未来有效性标签或 easy 真值作输入。

共完成 16 个新模型对象，包含 24 个森林，与 8 个已核验旧模型比较。
这是已暴露开发数据上的组件实验，不是新的神经动力学训练，也不是独立测试。

## 结果

三组决策代价改善分别为 -0.033689%、+0.461475%、0%。内部筛选 0/3。
第二组错误放行从 1 增至 19，因此不部署。这些百分比不是 ADE/FDE 提升。
训练完成与研究假设成立分开记录，旧部署模型和旧失败结论不变。

## 命令

在 `/Users/yangyue/Downloads/World` 使用原生 arm64 环境：

```bash
.venv-pytorch/bin/python scripts/run_m3w_moving_channel_specialist.py replay
.venv-pytorch/bin/python -m pytest tests/test_m3w_channel_specialist.py tests/test_m3w_boundary_audit.py tests/test_m3w_channel_specialist_reporting.py
```

首次完整流程是 `register`、`pilot`、`run`、`verify`。已完成阶段不可覆盖。
只有尚未完成的同一阶段才能加 `--resume`，恢复时会检查配置、特征、标签和行 ID。
`replay` 禁止拟合；检查点、逐行数据及缓存均留在本地，不在 GitHub。
缺少本地冻结资产时应停止并报告，不能用伪造或随机数据替代。

数据与模型路径：
`data/stage_cvpr2027_experiments/moving_boundary_audit_v1/` 和
`data/stage_cvpr2027_experiments/moving_channel_specialist_v1/`。

训练 CPU 4 线程、单进程、DataLoader workers=0。公共旧 runtime 消息中仍出现
forest_threads=1，实际注册并执行的 sklearn 配置为 n_jobs=4，不能混写。
pilot 约 3 秒、全部新拟合约 8 秒、全量重训检查约 12 秒；快是因为复用了本轮
已核验的拟合特征缓存，不是减少了约定的样本、折数或树数。前置几何/边界审计
约 113 秒、峰值内存约 7.67 GB；模型阶段不到 0.86 GB。无需新增 CREATE 作业。

结果用于决定下一步模型修改，不代表论文证据闭环完成。仍为 native obs8/pred12
图像坐标开发评价；不作米、秒、真实 3D 或 foundation 声明。Stage5C/SMC 不执行。
