# 完整排除校准场景的源数据实验

## 目的和边界

上一轮不是“模型没训练”，而是风险头低估了被选择样本的伤害。本轮要检验：
给固定模型使用独立于其完整训练链的源场景做校准，是否能保留收益并保护 easy。

这里的“独立”是每个内部划分的模型暴露边界，不代表12个开发场景变成了最终
独立测试。真正保留的模型选择、风险校准、最终确认和 DroneCrowd 组不打开。
历史 Stage35/37 结果不会因为本轮复算而恢复成无污染的正式结果。

## 数据链

每个源 fold 有4个场景。内部二分成2+2，训练相互不见预测目标场景的预测器；
其预测作为4个拟合场景上的收益/风险监督。已有的4场景预测器冻结使用，在
另外4个校准场景和4个外层读数场景推理。收益/风险头及其预处理也只见拟合组。

两个校准/外层分配方向都保留，三个训练种子都保留，不能根据结果挑好的那个。
本轮对比只能隔离同一新训练链中的校准作用；直接与此前8场景拟合结果比较，
会混入训练量和预测器差异。

## 环境与恢复

使用原生 arm64 `.venv-pytorch/bin/python`。CPU计算线程4，inter-op1，workers0。
所有训练入口在导入 Torch 前检查架构。每200步保存检查点，记录 PID 和UTC心跳。
预测器每50步、风险头每200步记录训练状态。试跑100步在总预算内恢复，不额外计数。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_nested_producers.py --prepare
.venv-pytorch/bin/python scripts/run_m3w_european_nested_producers.py --pilot
.venv-pytorch/bin/python scripts/run_m3w_european_nested_producers.py --train --resume --predict --replay
.venv-pytorch/bin/python scripts/run_m3w_european_nested_calibration.py --prepare
.venv-pytorch/bin/python scripts/run_m3w_european_nested_calibration.py --pilot
.venv-pytorch/bin/python scripts/run_m3w_european_nested_calibration.py --train --resume --calibrate --evaluate
.venv-pytorch/bin/python scripts/run_m3w_european_nested_calibration.py --verify --replay
.venv-pytorch/bin/python scripts/report_m3w_european_nested_producers.py
.venv-pytorch/bin/python scripts/report_m3w_european_nested_calibration.py
.venv-pytorch/bin/python scripts/report_m3w_european_calibration_transport.py
.venv-pytorch/bin/python scripts/plot_m3w_european_nested_calibration.py
.venv-pytorch/bin/python scripts/complete_m3w_european_nested_calibration.py
```

18个内部预测器各4,000步，共72,000步；54个收益/风险头各2,000步，共108,000步。
预测器保留旧采样规则：未来标签未知的行不贡献监督梯度，仍报告它们被抽中的次数。
收益/风险头只抽取已知标签。不要把两种计数混为“所有训练行都有标签”。

断线先检查 PID、心跳及最后检查点，确认原进程已结束后才恢复。活动锁不删除；
已有完成产物必须校验后复用。原始轨迹、预测缓存、权重留在本地，不提交GitHub。
每次训练前检查可用磁盘；低于10GiB停止新训练，不清除别的项目数据。

## 结果应该怎样解释

校准方案可以使用本轮内部校准组选择已登记的阈值，不能用外层结果再选。
“符合校准场景上的2%”不意味着对新场景有统计安全保证。样本有重叠；独立单位
是场景，不是轨迹窗口。这里不宣称 conformal 保证，也不改变原有2%限制。

必须同时看相对强运动对照的收益、最差场景 easy、零误差基线是否受损、干预率、
完整未来支持和条件场景bootstrap。只回退到CV的0提升不是成功。新头是否有效，
要等冻结后外层读数和重放完成，不能由训练loss、文件存在或脚本跑通推断。

图像像素、观察8步/预测12步、原始帧间隔12。不是t50，不是秒、米、物理安全、
人工gold、true3D或foundation。Stage5C和SMC不执行。

## 本轮已核验结论

18个预测器、54个收益/风险头全部完成，共180,000次更新。18份预测器和54份
头的检查点均精确重放；216份推理决策独立重构；196项针对性测试通过。

但神经模型没有超过同样受保护的阻尼基线：36组直接ADE比较的点估计全部为负，
其中34组条件区间完全为负。神经策略的观察性安全通过数从未校准的2/12提高到
两种校准各5/12；对应阻尼对照为12/12和11/12。这只是部分保护改善，不可部署。

校准组满足限制不等于外层场景满足限制：网格法的正向伤害约束在36/36校准
组合上通过，外层只有27/36。它与“净easy退化”不是同一个指标，不能混报。
下一步先定位预测器训练规模变化、源场景变化与收益排序误差，不继续利用这些
外层结果调阈值，也不打开保留确认集来挽救本轮失败。
