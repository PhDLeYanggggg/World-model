# 固定预算的新旧训练样本对照

环境使用项目内原生arm64 `.venv-pytorch/bin/python`，CPU4、inter-op1、workers0。
先保证上一轮producer-A交叉拟合缓存及原own-B缓存的哈希完整。
注册定义必须先提交；缓存、原始标注和模型checkpoint不上传Git。

空实验目录中的执行顺序如下，每个阶段正常结束后再进入下一阶段：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_donor_augmentation.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_donor_augmentation.py prepare
.venv-pytorch/bin/python scripts/run_m3w_moving_donor_augmentation.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_donor_augmentation.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_donor_augmentation.py verify
.venv-pytorch/bin/python scripts/run_m3w_moving_donor_augmentation.py evaluate
.venv-pytorch/bin/python scripts/report_m3w_moving_donor_augmentation.py
.venv-pytorch/bin/python scripts/run_m3w_moving_donor_augmentation.py replay
```

现有完成目录不用重跑训练，直接replay核对即可。报告和replay不能同时运行，
否则新增报告会触发“公共文件必须不变”的检查，即使预测数值完全一致。
本轮首次重放确实出现该并发报告问题，保留退出1记录；报告稳定后串行重验。
不要因观察超时或这个报告问题重新训练。

中断时先核实PID是否仍在运行；不要重复启动健康任务。
相同配置、代码、数据下，用 `train --resume` 或 `verify --resume` 恢复最近检查点。
每128步原子保存，每256步写心跳。心跳在本地
`data/stage_cvpr2027_experiments/moving_donor_augmentation_v1/heartbeat.json`。
保留5GiB磁盘、允许健康进程12小时；不要把长时间运行包装成已完成。

主训练24个模型共98,304次更新；完整重训另98,304次；恢复试跑另512次。
这不是32epochs。权重采样会重复访问部分样本，真实独立来源数和访问数量另记。
统计主结果的多seed及独立验证尚未完成，程序可复现不能替代它们。

信息筛选3/3不等于原安全gate通过：森林对照错误放行代价仍更低，暂不部署。
原2%门槛、支持要求、空对照NA、旧失败和主24组NA均不变。Stage5C/SMC不启用。
