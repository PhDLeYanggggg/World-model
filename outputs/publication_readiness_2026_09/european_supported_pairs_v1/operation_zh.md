# 有效事件配对修复：结果与复现

## 本轮实际完成

这次不是读取旧分数：重新训练了 36 个 Torch 风险头，共 72,000 次更新、3 个种子。
轨迹预测器、原风险头、数据划分和特征仍是经过哈希核验的旧资产，没有训练新轨迹模型。

唯一学习改动：先找到有定义的事件标签，再在同一场景、同一原始训练批次内配对。
此前是先配对再过滤，导致 easy-event 丢掉很多可用排序对。
模型大小、抽样、步数、原有损失、辅助损失权重和 2% 风险阈值均不变。

## 成功与失败

- 配对效率确实改善：神经 easy 有效训练配对从 167,291 增至 541,866，约 3.24 倍。
- 9 个神经 easy 头的固定训练批次总 loss 都下降，8 个排序 loss 下降；不是验证集结果。
- 但固定切换数量后，排序优势仍不稳定。仅增加有效配对不足以解决问题。
- 与同等保护的阻尼基线相比，18 个总体 ADE 对照全部为负，17 个区间明确为负。
- 神经策略最差 positive-easy 退化为 0.0606%，低于 2%，但 11 个视图仍伤害 CV 原本零误差的样本。
- 阻尼 easy 的准确率也受到这项修改的负面影响，不能把差距缩小说成神经模型变强。
- 不升级部署，没有证明新神经动力学贡献，仍未达到投稿候选证据要求。

另做了一个不使用真实数据的数值反例：逐样本 H/(B+H) 的排序可以与部署需要的
E[H|x]/E[B|x] 排序相反，现有排序损失甚至可能偏好低估危险状态。
这说明目标定义存在结构性风险，不证明它是实测失败的唯一原因。
下一轮应单独检验目标定义，而不是继续改配对花样或放宽阈值。

## 验证与运行

36 个检查点回放、216 个完整视图和 36 个旧对照均复现。
另外一套逐项排序、坐标误差及 bootstrap 算法完成核验。
247 项相关测试通过，覆盖 39 个文件；不是全部历史仓库测试。
真实试训已断点续跑至完整预算；所有必要进程都已结束。

```bash
cd /Users/yangyue/Downloads/World
.venv-pytorch/bin/python scripts/run_m3w_european_supported_pairs.py --prepare
.venv-pytorch/bin/python scripts/run_m3w_european_supported_pairs.py --audit
.venv-pytorch/bin/python scripts/run_m3w_european_supported_pairs.py --train --resume --replay --decide
.venv-pytorch/bin/python scripts/run_m3w_european_supported_pairs.py --evaluate
.venv-pytorch/bin/python scripts/run_m3w_european_supported_pairs.py --verify
.venv-pytorch/bin/python scripts/diagnose_m3w_ratio_ranking_estimand.py
.venv-pytorch/bin/python scripts/report_m3w_european_supported_pairs.py --complete --figures
```

复现需要本地原始资产、冻结预测缓存和检查点。GitHub 轻量结果本身不包含它们。
已完成的阶段会核验身份再复用；不要删除检查点或启动重复任务。
中断后在原目录用 `--resume` 恢复。哈希不一致应调查，不能忽略。

本轮原生 arm64、CPU4、inter-op1、workers0，无多进程 DataLoader。
纯拟合累计约 111.802 秒，不含准备、推理、评估和核验；完整训练、回放、冻结阶段约
207.818 秒。不是速度比较实验，也没有因慢缩减规模。无需 CREATE 训练资源。

## 证据边界

全部 12 个 European Squares 场景都已开放用于开发。每轮四场景拟合、八场景完整
上游链排除，但这不能恢复独立最终测试资格。当前是检测轨迹像素坐标、观察 8 步预测
12 步、raw stride 12，不是历史 t+50、秒、米、人工 gold、物理安全、true 3D 或 foundation。
不重新认证历史 Stage37，不执行 Stage5C，不启用 SMC，保留集继续关闭。
