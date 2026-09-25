# 排序辅助训练：本轮结果与复现

本轮不是重复读取旧报告，而是真正训练了36个 Torch 风险头，共72,000次更新。
轨迹预测器、原风险头和数据划分是经过哈希校验的旧资产，没有重新训练轨迹模型。

## 结果

- 新增“同场景内风险排序”损失，模型容量、抽样、步数和风险限制不变。
- 神经模型部分总体指标比原风险头改善，但固定切换数量后排序没有稳定变好。
- 与同等保护的阻尼基线比较，18个总体ADE点估计全部为负，17个区间明确为负。
- 18个神经策略均满足positive-easy百分比限制，但12个仍伤害CV原本零误差的样本。
  另外6个没有这类样本，不代表证明保护成功。
- 新阻尼策略有3个easy退化失败，最差4.9464%，因此也不能只按准确率升级部署。
- 216个完整视图复现；234项相关测试通过，不是全历史仓库测试。

有效配对不足值得下一步修复：easy-event训练平均每批只有约9.29个有效神经排序对，
all-event约130.85个。这来自先配对再删除无定义事件标签的设计。它可能造成训练噪声，
但尚未证明是唯一原因。下一轮应只改变训练配对方式，不能根据当前读出挑阈值。

## 复现命令

需要已有本地原始资产、冻结预测缓存和检查点，GitHub轻量结果包本身不能重新训练。
在项目根目录使用原生arm64环境：

```bash
cd /Users/yangyue/Downloads/World
.venv-pytorch/bin/python scripts/run_m3w_european_ranked_hurdle.py --prepare
.venv-pytorch/bin/python scripts/run_m3w_european_ranked_hurdle.py --train --resume --replay --decide
.venv-pytorch/bin/python scripts/run_m3w_european_ranked_hurdle.py --evaluate
.venv-pytorch/bin/python scripts/run_m3w_european_ranked_hurdle.py --verify
.venv-pytorch/bin/python scripts/report_m3w_european_ranked_hurdle.py --complete --figures
```

已完成的头会先检查身份和文件哈希再复用。中断时从原目录加`--resume`继续，
不要删除检查点或者另开重复任务。验证阶段会重新提取检查点预测、核对抽样、
重算全部决策和指标，并用另一套排序和坐标误差计算验证。任何哈希不匹配必须调查。

本轮CPU4、inter-op1、workers0。所有头的纯拟合时间合计149.734秒，不含数据组装、
推理、评估和报告；完整训练回放决策阶段254.194秒。没有因慢缩减实验，也没有卡死。
所有必要进程均已结束。日志和恢复状态位于私有`european_ranked_hurdle_v1`目录。

## 边界

这是已开放的European Squares开发场景、检测轨迹像素坐标、观察8步预测12步、raw
stride12。不是历史t+50，不是秒、米、人工gold、物理安全、true3D或foundation。
所有场景均已参与开发，bootstrap和三个种子不能恢复独立最终测试资格。
不升级部署，不重新认证历史Stage37，不执行Stage5C，不启用SMC，保留集继续关闭。
