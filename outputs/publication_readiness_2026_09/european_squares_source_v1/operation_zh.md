# European Squares 训练数据：运行与恢复

本数据集的用途已经在任何预测误差读取前冻结。当前只允许使用
`source_training`；模型选择、风险校准、最终确认三个角色仍关闭。
这里的“完整构建”指完整执行已登记的每录像最多256个查询帧方案，
不指物化全部原始窗口，也不指 medium/full 模型训练完成。

## 环境与入口

在 `/Users/yangyue/Downloads/World` 中使用本机原生 arm64 环境。
本步骤是单进程 NumPy 数据构建，不是 Torch 训练或 GPU 成功证明。
不需要 CREATE，也不会触碰 simulation 工程或现有作业。

```bash
.venv-pytorch/bin/python scripts/freeze_m3w_european_squares_roles.py
.venv-pytorch/bin/python scripts/build_m3w_european_squares_source.py --pilot
.venv-pytorch/bin/python scripts/build_m3w_european_squares_source.py
.venv-pytorch/bin/python scripts/build_m3w_european_squares_source.py --verify
```

第二条仅是第一条录像的真实试跑。第三条必须完成所有163条训练录像；
第四条重新解析全部训练原始数据并逐数组对比，不只是查看“完成”文件。
冻结角色脚本会重新验证依赖和角色，重复运行必须得到相同角色文件。

## 恢复和日志

私有目录：
`data/stage_cvpr2027_experiments/european_squares_source_v1/`

`heartbeat.json` 保存当前 PID、录像序号、状态；`events.jsonl` 保存运行记录。
每录像的缓存和 receipt 单独落盘。中断后重新运行构建入口即可校验已有
文件并继续，不需要删除缓存。文件锁阻止重复构建。依赖身份改变、缓存
哈希不符或磁盘保留空间不足会报错，不能通过改 receipt 绕过。

## 输入与标签

每录像的 `input_*.npy` 保存过去坐标、过去框、因果差分速度、历史 mask、
当前可见 agent、仅由过去决定的 target eligibility 和 CV rollout。
`query_offsets` 将同一查询帧的所有可见 agent 放在一起。

`label_future_xy.npy` 和 `label_future_valid.npy` 是分开的监督标签，
不得拼接进网络输入。任何新 dataloader 都必须用固定输入白名单。
没有未来标签的目标仍保留，训练 loss 使用 mask，不能伪造零误差。
不能把缓存里未来可用长度或整段 track 长度作为推理特征。

坐标是未经平滑的检测框中心 image pixels。8步观察、12步请求预测、
raw stride12 是这里明确登记的外部索引任务，不代表与 SDD 等时长，
不代表秒、米、真实3D或人工 gold。邻居距离只能作 image-plane proxy。

## 后续训练边界

下一步是在12个训练地点内固定 source-only 的拟合与内部交叉拟合，
检查误差、增益/伤害目标和简单对照是否可学。先登记比较、预算和种子，
再打开训练侧误差；不要先查看保留地点成绩再定训练方案。
保留角色需要单独的冻结实验和 producer 身份后才能读取预测结果。
一旦支持不足，保留 fallback，不改2% easy限制或零参考伤害限制。
Stage5C、SMC和论文正式提交均不在本入口范围内。

代码、角色清单和轻量报告可提交 Git；原始压缩包、这些缓存、模型权重、
第三方数据和虚拟环境不得提交。
