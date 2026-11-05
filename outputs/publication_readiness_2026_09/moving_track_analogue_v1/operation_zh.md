# 原规则下的跨来源历史检索实验

## 做了什么

按照“保持原规则，先改模型”执行。没有加入恒等回退认证，没有提高 2% 门槛，
没有删除支持要求，也没有把 NA 改成通过。

这次改变的是风险预测方法：对每条候选预测，从其他来源寻找相似的因果历史，
每个来源/agent 身份桶只保留一个窗口，再用 32 个桶估计风险。整段查询来源
均不能作为参考，避免重叠轨迹窗口被当作额外证据。身份桶不是已核实的独立行人。
与旧森林、训练集平均风险和打乱匹配的对照比较，全部 48,061 条已知结果记录完成。

正确匹配确实优于打乱匹配，但三个设置相对旧模型的决策成本改善分别为
-4.481672%、-121.691096%、-2.568241%。错误放行从 0/1/0 变为 107/723/521。
所以内部筛选为 0/3，不部署。这里的成本百分比不是 ADE/FDE 提升。
这是风险组件实验，不是新的 Transformer/JEPA 动力学训练，更不是独立测试。

## 路径和命令

在 `/Users/yangyue/Downloads/World` 使用原生 arm64 环境。
首次流程依次是 `register`、`prepare`、`pilot`、`run`、`verify`、`replay`：

```bash
.venv-pytorch/bin/python scripts/run_m3w_moving_track_analogue.py replay
.venv-pytorch/bin/python -m pytest tests/test_m3w_track_analogue.py tests/test_m3w_track_analogue_reporting.py tests/test_m3w_channel_specialist.py tests/test_m3w_channel_specialist_reporting.py tests/test_m3w_boundary_audit.py -q
```

已经完成的阶段不可覆盖。若某阶段实际中断，只恢复同一阶段，例如：

```bash
.venv-pytorch/bin/python scripts/run_m3w_moving_track_analogue.py run --resume
```

恢复会核对注册、配置、缓存、逐行 ID 和已保存的参考。`replay` 禁止重新拟合
及距离检索，只从已保存参考重建预测和指标，并检查公共文件没有被改写。
首次只读 replay 还会在私有目录保存完成回执；重复调用已完成阶段会明确拒绝，
不能通过删除原回执伪装成 fresh run。报告生成脚本只渲染结果，不训练或选模型。

本地私有资产位于：
`data/stage_cvpr2027_experiments/moving_track_analogue_v1/`。
原特征缓存位于 `data/stage_cvpr2027_experiments/moving_boundary_audit_v1/`。
缺少原数据或检查点时，GitHub 上的轻量报告不能独立重建真实结果，应如实报缺失。

元数据核验耗时 17.698 秒；pilot 0.742 秒；全量检索 37.138 秒；
换分块完整重算 32.254 秒。全量峰值 RSS 约 1.52 GB。快是因为重用已核验特征，
不是缩减样本。全部八折精确重现；18 项针对性测试通过，未重跑历史全测试集。
CPU4/inter-op1/workers0；距离核按单进程分块运行。旧运行日志中的 forest_threads
字段不代表本次又训练了森林。每 16 块和每折保存检查点，持续记录心跳。

本机足够，无新增 CREATE 作业。旧 CREATE 回执不能证明当前队列状态。
数据、缓存、检查点、原始图像不上传 Git；代码、配置、轻量结果与报告安全提交。
当前仍是 native obs8/pred12 图像坐标开发任务，不作米、秒、true3D、foundation
或人工 gold 声明。Stage5C/SMC 不执行，论文尚未达到投稿准备完成条件。
