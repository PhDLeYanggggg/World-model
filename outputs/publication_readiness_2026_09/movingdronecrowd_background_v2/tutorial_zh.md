# 新来源图像排重：结果与复现

## 本轮完成了什么

已完成全部 120 个 MovingDroneCrowd++ 片段的三视图来源检查，合计
360 个新视图，并复用经校验的 336 个 DroneCrowd 缓存视图。新来源内部
7,140 对、跨来源 13,440 对，合计 20,580 对全部保存。

原匹配进程实际跑了约 69 分钟，之后在汇总字典时因重复字段退出。
保留了退出码 1 和全部检查点，没有重跑或篡改原定义。单独注册的汇总
修复入口用约 3.92 秒恢复汇总，不能把这段时间写成完整匹配耗时。

随后正式复核用约 43.10 秒重提所有 360 个视图特征，核验全部旧缓存、
匹配记录和分组，再重新算固定的 50 对，全部一致。额外的只读复核约
42.56 秒，也全部一致。这里没有第二次完整下载或第二次全量匹配。

## 结果如何理解

- 跨来源：0 强候选、0 模糊候选、13,440 对不支持匹配。
- 新来源内部：2 强多视图候选、29 模糊候选、7,109 对不支持匹配。
- 保留全部发布方分组和旧保护分组，共 89 个保守组件；不是 89 个独立地点。
- 发布方训练集的 64 个片段仍为候选，含 136,295 个完整原生 8/12 窗口。
- 同一候选人口中还有 58,991 个历史完整、未来不完整的 agent 实例。
  不能只保留未来完整的人作为推理邻居，从而泄露标签可用性。

未匹配不等于证明不同地点。213 对同发布方标签中，仅 31 对在此稀疏视图
规则下有支持；发布方标签也不是精确视野真值，不能拿它估算地理召回率。
本轮未覆盖 DUT 图像，也没有完成训练用途或完整 producer-chain 接纳。
所有 112 个 DroneCrowd 片段仍整源保留为 confirmation，不挪入训练。

## 日常只读复核

在项目目录 `/Users/yangyue/Downloads/World` 使用原生 arm64 环境：

```sh
.venv-pytorch/bin/python scripts/check_m3w_movingdronecrowd_background.py
.venv-pytorch/bin/python scripts/report_m3w_movingdronecrowd_background.py --version2
```

第一条只读检查冻结定义、源数据身份、360 个重提特征、336 个旧视图、
20,580 个保存结果及分组，并重新匹配原先固定的 50 对。它不下载数据、
不写应用结果文件，不修改已经完成的执行记录。第二条只从已核验结果
生成标量报告，不是重新计算模型成绩。完整复现仍需要私有缓存和源材料，
仅 GitHub 轻量文件不足以独立冷启动，这是当前复现限制。

## 中断和错误恢复

完成阶段不可覆盖。不要删除成功结果来制造 fresh_run。
若是匹配尚未完成且原进程已确认退出，才使用原命令恢复：

```sh
.venv-pytorch/bin/python -u scripts/audit_m3w_movingdronecrowd_background_v2.py run --resume
```

但本轮全部匹配已经完成，原汇总代码有明确重复字段错误，不应再走这个
入口重试汇总。本轮的一次性恢复与验证命令是：

```sh
.venv-pytorch/bin/python -u scripts/finalize_m3w_movingdronecrowd_background_v2.py recover
.venv-pytorch/bin/python -u scripts/finalize_m3w_movingdronecrowd_background_v2.py verify
```

二者均已完成，再运行会拒绝覆盖；日常核对应使用上面的只读检查器。
原日志、heartbeat、pair 文件保留在私有
`data/stage_cvpr2027_experiments/movingdronecrowd_background_v2/`。
先核对进程与日志，再决定是否恢复，不能因看不到旧工具句柄就重复启动。

本地一次约 69 分钟、匹配峰值约 674 MB，适合本机，未使用 CREATE 新训练
任务。保留单 OpenCV 线程、原 10 GiB 磁盘余量和 12 小时边界，不因慢缩小
样本。原生视图/annotation step 未获秒级或米制校准；图像变换不是地面
homography。没有新模型训练、预测提升、人工 gold、true 3D、foundation、
Stage5C 执行或 SMC 成果。科研目标仍未完成。
