# 同时刻联合选择：复核、调用与恢复

## 本轮做了什么

工作目录：`/Users/yangyue/Downloads/World`。
固定已有18个成本预测头、六个物理候选及一个神经预测，只比较决策方式。
没有新训练神经网络，也没有新的训练loss曲线；优化器目标是固定预测成本下的联合分配。
主比较逐个当前查询严格匹配切换数量，不是只匹配整表平均切换率。

完整72组和另一个进程的完整重放已经完成。拟合收益提高，但原定风险要求失败；不部署。
这些是已经暴露的SOURCE拟合数据，不是新的独立测试。任务观察8步、请求未来12步，
raw stride12，不是补充raw t+50，也不能写成米、秒、人工gold、true3D或foundation。

## 已完成的真实运行

|阶段|PID|用时|结果|
|---|---:|---:|---|
|真实128查询试跑|18870|21.04秒|含准备，恢复与直接运行相同|
|完整72组分配和评价|19894|729.54秒|全部完成，峰值10.92GB|
|单独进程完整重放|22148|741.43秒|预测、动作、指标全部相同|

PID是历史记录，不代表当前进程仍在运行。源权重继续保存在原先
`data/stage_cvpr2027_experiments/european_admission_cost_v1/train/`。
本轮私有动作检查点、heartbeat和日志位于
`data/stage_cvpr2027_experiments/european_frame_action_allocation_v1/`。
公开注册文件记录全部源权重路径/hash；不把权重、原始数据或缓存提交GitHub。

## 只验证公开聚合结果

```bash
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_frame_action_allocation_v1 --verify
```

这只核对72个聚合JSON的完整字节，不重新推理，不需要把聚合数字当独立科学重现。
需要展开聚合JSON时使用 `--restore`。750,718B压缩包只含聚合指标，不含原始轨迹或权重。

## 有合法私有数据及权重时重放

```bash
.venv-pytorch/bin/python scripts/run_m3w_frame_action_allocation.py verify
.venv-pytorch/bin/python scripts/report_m3w_frame_action_allocation.py
```

前者实际重做全部推理、分配和评价；后者从已验证结果生成报告和图表，不是再次科学重算。
程序检查注册代码、模型hash、行对齐和固定源对照。正式结果不可覆盖。
只有公开仓库、没有私有源数据和既有权重，不能声称成功重训或完整重放。

## 尚未完成运行的恢复

```bash
.venv-pytorch/bin/python scripts/run_m3w_frame_action_allocation.py run --resume
```

只对没有完整完成凭据的中断任务使用。本轮已完成，不能再用该命令覆盖结果。
每250查询原子保存一次动作进度，恢复时检查同一预测、可用动作、当前帧和行ID。
完成的组保留；未完成的组继续。12小时、10GiB剩余磁盘保护和单进程锁防重复任务。

运行应使用arm64 `.venv-pytorch`、CPU线程4、interop1、workers0，不用x86_64 Conda。
本轮实测本地资源充足，因此没有向CREATE提交或复制任务。未来若需要CREATE，先核实
实时调度器和已有作业，不能用旧SSH故障或旧job ID推断当前状态。

## 程序调用边界

`src.world_model.m3w_frame_action_allocation.allocate` 接受固定的七候选五矩预测、
因果可用性、录像ID、当前raw frame和行ID，返回四种策略的动作及诊断信息。
它不接受future endpoint或未来标签。目标标签只进入evaluation模块。
这是研究API，不是已部署模型接口；原有部署不因本轮均值改善而改变。

后续神经使用取证在独立目录 `european_frame_allocation_attribution_v1`，读取冻结动作，
不改变选择。删除已使用的神经动作只是消融，不等于重新优化一个纯物理联合策略。
该区分必须保留。Stage5C未执行，SMC未启用。
