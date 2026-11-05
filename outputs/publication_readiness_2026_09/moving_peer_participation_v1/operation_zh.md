# 复现与恢复

本轮按“保持原规则，先改模型”执行。任务是原生 obs8/pred12 图像坐标，
不是秒或米。使用已暴露的 SOURCE/own-B 开发数据；没有打开新的保留测试集。
原2%风险、easy-net及支持要求不变，空支持仍为NA，没有恒等回退认证。

## 环境与文件

在项目根目录使用 `.venv-pytorch/bin/python`，入口拒绝非原生arm64。
本轮为真实PyTorch CPU训练，线程4、interop1、workers0，没有NumPy替代训练。
每100步保存checkpoint及heartbeat，进程锁禁止重复运行。
私有文件在 `data/stage_cvpr2027_experiments/moving_peer_participation_v1/`，
数据、模型权重、缓存和环境不上传GitHub。

本轮新增六个模型，每个1000步；另外六个旧对照是cached_verified，不是
重新训练。500步试跑验证两种新模型的断点恢复及旧模型100步精确复现；
另做1000步完整重复训练。全部实际新增更新数为7500。

## 原始运行顺序

先提交协议、代码和注册文件，再顺序执行：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_peer_participation.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_peer_participation.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_peer_participation.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_peer_participation.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_peer_participation.py verify
.venv-pytorch/bin/python scripts/diagnose_m3w_moving_peer_participation.py
.venv-pytorch/bin/python scripts/report_m3w_moving_peer_participation.py
```

这些阶段已完成。不要把上述原始顺序在已完成目录中重新执行以覆盖结果。
完成凭据存在时会拒绝重跑。仅在进程确已退出且某阶段未完成时，对该阶段
加 `--resume`。不要因观察超时重复启动，不要删除checkpoint。
报告、诊断和只读复核共用同一锁，必须顺序运行。

## 已完成结果只读复核

```sh
.venv-pytorch/bin/python scripts/replay_m3w_moving_peer_participation.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_peer_participation_risk.py tests/test_m3w_peer_participation_reporting.py
```

只读复核重新计算12个组合的初始、最终指标及错误代价分解，核对公开文件
未变，不重新训练，也不新增科学评价。读取旧权重并精确复现不能标成新训练。

## 读结果

`result_summary.json`包含全部四组对照及执行信息，`fitting_diagnosis.json`
包含来源切片、邻居是否回退的切片，以及共享初始模型中的表示差异。
`analysis.json`明确记录额外nested-C/H为not_run：固定主要对照未过内部
检查。这个内部检查不是科学安全认证，也不能替代原来的风险评价。

主对照的决策代价改善为-0.0433%、-4.1076%、+0.6746%，不是测试ADE。
不以次要组合里的局部改善换掉主要结果。旧失败保留，不升级部署，
Stage5C/SMC关闭。当前仍不具备投稿准备完成的证据。
