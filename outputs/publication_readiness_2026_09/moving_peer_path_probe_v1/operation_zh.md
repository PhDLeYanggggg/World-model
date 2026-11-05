# 运行与复现记录

本轮保持原评价规则，不加入恒等回退认证。当前任务是原生观察8步、预测12步的
干预风险信息实验，不是旧SDD的t+50实验，也不是新世界动力学模型。

## 本轮实际执行

1. 固定协议、代码和模型参数先提交为`fec5f2be90fb6f76a18235c5d3870e271b8f01de`。
2. 真实数据64棵树试跑并精确重复：PID39075，正常退出，41.054秒，峰值内存
   5,917,310,976字节；单次完整森林拟合2.210秒。
3. 完整48项源留出拟合：PID39226，正常退出，60.167秒，峰值内存
   6,350,716,928字节。其中24项ExtraTrees、24项ridge；8项ridge是旧对照的精确复现。
4. 全部48项重新拟合、模型结构/系数/预测/指标核验：PID39477，正常退出，
   60.569秒，峰值内存6,521,110,528字节。全部精确一致。
5. 33项针对性回归测试通过，2.88秒。未重跑整个历史测试套件。
6. 对照图已查看；无截断、文字重叠或空白。图展示负结果和错配对照，未仅选正结果。
7. 最终只读复现PID39789正常退出，43.190秒，峰值内存6,474,055,680字节。
   拟合入口被禁止调用，48项保存预测全部精确复现，当时67个公共文件完全未变。
   日志沿用的`probe_fitted`状态名称不代表这次重新训练；该次只有加载和预测核验。

这不是因为慢而提前停止或降低规模。固定48项全部完成；当前小规模信息探针很快。
检查点按拟合单元保存，已有单元可恢复，不把观察超时当成训练失败。

## 命令

在项目根目录使用原生arm64环境；不得用x86 Conda/Rosetta。

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_peer_path_probe.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_peer_path_probe.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_peer_path_probe.py run
.venv-pytorch/bin/python scripts/run_m3w_moving_peer_path_probe.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_peer_path_probe.py
.venv-pytorch/bin/python scripts/replay_m3w_moving_peer_path_probe.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_peer_path_probe.py tests/test_m3w_peer_path_reporting.py tests/test_m3w_candidate_neighbor_probe.py tests/test_m3w_candidate_neighbor_reporting.py tests/test_m3w_peer_participation_risk.py tests/test_m3w_peer_participation_reporting.py
```

已完成阶段不可覆盖；现在应使用`replay`加载已有模型、重算预测并核对报告哈希，
不重新拟合、不修改公共结果。只有实际中断且没有完成凭据的`run`才能使用
`run --resume`。缺失或变更的输入、模型、协议不自动修补。

Torch线程4、interop1、DataLoader workers0。ExtraTrees实际保存参数为`n_jobs=4`；
继承的通用runtime日志中`forest_threads=1`是旧字段，不是当前森林参数。
日志和逐模型检查点在忽略的私有目录；无数据、缓存或模型权重上传GitHub。

## 如何解释结果

1/3通过的是本轮内部继续研究筛选，不是原科学安全gate。另两组失败，不能选择
最好的一组就宣布整体成功。第三组风险预测误差改善约5%，但38,579个已知结果候选
仍全部被拒绝，决策代价没有改善。第二组有害接纳从1增加到10，也明确保留。

`analysis.json`哈希：
`747954f861fee9888659ed88acaa69b70707aa6691af3e70004bf6e53a7c0a91`。
精确重复只证明计算可复现，不等于独立统计确认。没有新增测试集读数，没有部署升级。
本轮附录未并入已编译的v9论文PDF；不能把新文字当成已经重新排版验证的论文。
