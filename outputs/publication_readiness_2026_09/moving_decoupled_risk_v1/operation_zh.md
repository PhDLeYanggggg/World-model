# 本轮实验运行与恢复

## 不变的规则

原来的2%风险、easy退化和支持量规则全部保留，不加入恒等回退认证。
缺失支持和空对照仍为不可估，旧失败不改写。本轮是观察8步、预测12步的
图像坐标研究，不是SDD raw-frame t50结果，不提供米或秒的解释。
保留的独立场景和H不开放，Stage5C和SMC不执行。

## 环境与命令

在项目根目录使用原生arm64的`.venv-pytorch/bin/python`。
入口会在导入数值库前拒绝macOS上的x86_64/Rosetta。
固定CPU4计算线程、1个互操作线程、DataLoader workers0，不做资源探测。

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_decoupled_risk.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_decoupled_risk.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_decoupled_risk.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_decoupled_risk.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_decoupled_risk.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_decoupled_risk.py
```

协议、代码和注册哈希须在训练前提交，入口会核验。已完成的阶段拒绝覆盖。
只有真正中断的原配置才能续跑，例如：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_decoupled_risk.py train --resume
```

不要在已有训练进程仍运行时另起任务。进程锁阻止同实验重复运行。
模型每100步保存完成更新后的原子检查点，并记录心跳；中断时不能把半次
参数更新覆盖到检查点。恢复包括模型、优化器、抽样器、随机状态和训练轨迹。
12小时运行保护、至少5GiB剩余磁盘、新私有产物上限512MiB。

## 核查完成状态

训练完成需要九份fit记录和training_freeze.json，不能只看日志里出现loss。
每份必须有2000次实际更新，两条相关编码器确实更新，参数有限，输入和标签
哈希匹配。最大上下文三个结构都有50+50与连续100步的实际恢复一致性检查；
第一组分离结构完整2000步重跑一次。最终还需全部初始/最终拟合指标精确重放。

evaluate成功退出不一定代表跑了开发集预测。如果预先固定的拟合筛查失败，
它仅写not_run及原因。不得把这种退出状态当作测试通过。
若允许额外读出，也仅限已暴露的SOURCE/nested-C开发角色，不是独立测试。

私有检查点、心跳和预测缓存位于
`data/stage_cvpr2027_experiments/moving_decoupled_risk_v1/`，不上传Git。
公开协议、轻量数字和报告位于本目录。旧CREATE记录仅为历史资产，
本轮本机试跑支持本地执行，没有新提交HPC作业。
