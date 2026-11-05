# Producer-A 交叉拟合复现

使用仓库内原生 arm64 `.venv-pytorch/bin/python`，CPU4、inter-op1、workers0。
所有命令在项目根目录执行。旧文件与数据哈希须一致，不能用新数据覆盖原实验。

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_producer_crossfit.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_producer_crossfit.py prepare
.venv-pytorch/bin/python scripts/run_m3w_moving_producer_crossfit.py pilot
.venv-pytorch/bin/python scripts/verify_m3w_producer_crossfit_pilot.py
.venv-pytorch/bin/python scripts/run_m3w_moving_producer_crossfit.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_moving_producer_crossfit.py forecast
.venv-pytorch/bin/python scripts/run_m3w_moving_producer_crossfit.py replay
.venv-pytorch/bin/python scripts/report_m3w_moving_producer_crossfit.py
```

上述是空实验目录的历史执行顺序，不要在已经完成的目录重新运行 pilot。
原 pilot 检查器会因恢复路线多出第20步日志退出1；此失败记录保留。
随后的专用验证器核实所有训练状态、共同日志和推理输出逐项完全一致；
没有删除日志，也没有重新训练来覆盖差异。完整模型只使用原40步恢复前缀。
现有结果只需运行 replay 和 reporter，不必重复训练。

中断后检查 `data/stage_cvpr2027_experiments/moving_producer_crossfit_v1/heartbeat.json`
与 PID 是否仍运行。不要重复启动健康任务。使用相同代码、配置和数据加
`train --resume` 从最近原子检查点恢复；不选取中途效果最好的 checkpoint。
每200步保存、每50步写心跳，健康进程允许12小时。磁盘保留5GiB。

缓存、原始标注、checkpoint 留在本地，不上传Git。新缓存仅在原预测器A内部
交叉拟合，不能据此把外层C/H或其他受保护来源移入训练。原2%规则、空支持NA、
旧失败、原24组主估计NA全部保留。训练完成不等于风险头提升或投稿准备完成。
