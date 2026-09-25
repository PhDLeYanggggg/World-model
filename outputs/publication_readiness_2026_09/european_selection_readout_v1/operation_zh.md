# 六个模型选择场景：运行与恢复说明

本轮已经完成真实转换、冻结推理和评估，没有重新训练神经网络。注册提交为
3d8ee3d1；预测与动作冻结提交为c78fc812。两次都在后续读出之前推送。
这是模型选择证据，不是最终独立测试，也没有升级部署。

## 环境与安全边界

在项目根目录使用原生arm64环境`.venv-pytorch/bin/python`，计算线程4、interop1，
没有DataLoader多进程。入口在导入Torch前拒绝Mac上的x86_64运行。输入仅接受过去
几何、历史和当前位置；标签另存。不能用系统默认的旧x86_64 Conda替代。

原始数据、模型权重和中间数组均留在`data/stage_cvpr2027_experiments/`和
`external_data/`的忽略路径。不要把这些目录加入Git。磁盘低于10GiB时程序停止
写入并保留已完成记录，而不是降低样本规模伪称完成。

## 复现顺序

以下命令复用相同版本的冻结模型；再次评估已经开放的场景不能算新独立证据。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_selection_readout.py --phase preflight
.venv-pytorch/bin/python scripts/run_m3w_european_selection_readout.py --phase build
.venv-pytorch/bin/python scripts/run_m3w_european_selection_readout.py --phase decide
.venv-pytorch/bin/python scripts/run_m3w_european_selection_readout.py --phase evaluate
.venv-pytorch/bin/python scripts/report_m3w_european_selection_readout.py
.venv-pytorch/bin/python scripts/diagnose_m3w_european_selection_readout.py
.venv-pytorch/bin/python scripts/verify_m3w_european_selection_readout.py
```

`register`只用于首次冻结，不能用来覆盖已有锁。打开新用途数据前，注册代码、
模型清单和锁必须已经提交并推送。评价前完整预测冻结也必须提交；本轮已推送。

每段录像、每组推理和每组评价都有哈希凭据。中断后重跑同一阶段，已完成部分
核对哈希后复用，不重新选阈值。模型、代码或划分不匹配会报错，不能删除凭据
强行继续。私有目录中的`heartbeat.json`和`events.jsonl`记录PID与阶段进展。

## 如何解读

新增介入在36组配置中均改善相对旧控制器的平均ADE，但只有29组满足easy约束。
全体风险目标下存在明显局部伤害；easy风险目标18组都守住了easy，但仍不是
每组都超过强运动基线。因此不部署新策略，不打开校准集来反复调参。

完整结果、三种子汇总、失败场景、缺失标签和零误差基线伤害都保留。
统计单位为六个locality group，而不是38,102条重叠窗口或36个重复配置。
12个风险校准场景和6个最终确认场景继续封闭。Stage5C、SMC都未执行。

CREATE本轮仅做了经授权的只读队列检查，返回成功；没有提交远程作业或修改
远程文件。本轮推理规模适合本机。远程M3W资产路径未验证不等于资产不存在。
