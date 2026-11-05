# 候选轨迹与邻居关系诊断

本实验不改变原有2%评价规则，也不增加恒等回退认证。
当前是观察8步、预测12步的图像坐标任务，不是秒或米。

## 已完成的实验

三个训练上下文，八个留一来源划分，十六次固定岭回归探针拟合。
增加24个因果邻居关系特征后，三组平均风险预测误差都变差。
这是模型输入假设的负结果，不是新神经模型训练成功，也不是部署通过。

## 复核已完成结果

在项目目录使用原生arm64环境：

```bash
.venv-pytorch/bin/python scripts/replay_m3w_candidate_neighbors.py
.venv-pytorch/bin/python scripts/report_m3w_candidate_neighbors.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_candidate_neighbor_probe.py tests/test_m3w_candidate_neighbor_reporting.py
```

第一条命令重算探针系数、预测和指标，与本地缓存逐项比较，不改公开结果。
第二条只从已验证指标生成表格与图。第三条是针对性代码检查，不是科研成功证明。
原始运行入口的run/verify阶段已有完成记录，不应覆盖或重复启动。

CPU4、线程间并行1、数据加载workers0。没有CREATE新任务。
本地逐行特征、探针系数和预测保存在data下，不提交GitHub。
公开报告不包含原始轨迹、图像、视频或大模型文件。

缺失支持仍是不可估，旧实验失败保持不变；Stage5C和SMC均未执行。
