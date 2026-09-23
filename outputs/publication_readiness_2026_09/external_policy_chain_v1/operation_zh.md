# 冻结推理流程：使用与复核

## 这次完成了什么

现在已有一个完整的数值入口：过去八步观测 → 固定轨迹预测器 → 增益/伤害
估计 → 独立或联合选择 → 基线回退。它是研究接口，不是已经通过独立风险
校准的可部署模型。本轮没有新增训练，没有读取外部保留集的预测结果。

模型权重来自上一轮的本地源数据训练。六个预测器和十二个成本头的路径、
哈希及代码依赖保存在同目录的 `policy_manifest.json`。权重本身没有上传。
不要修改这些文件后仍把它们称为同一个冻结版本。

## 环境与复现

在 `/Users/yangyue/Downloads/World` 使用原生 arm64 环境，不使用 x86_64 Conda。
本次脚本配置四个计算线程、一个 interop 线程和零 DataLoader worker。
本地耗时合理，无需占用 CREATE；没有触碰 simulation model 的作业。

```sh
.venv-pytorch/bin/python scripts/freeze_m3w_external_policy_chain.py --audit-only
.venv-pytorch/bin/python scripts/freeze_m3w_external_policy_chain.py --verify
.venv-pytorch/bin/python scripts/summarize_m3w_external_chain.py --verify
```

第一项检查输入与依赖；第二项重新加载模型并核对完整决策；第三项重算
轻量汇总。都不是新的模型训练，也不会算出外部准确率。

若 `--run` 中断，重新运行同一命令会复用已经完成的模型/成本头组合，从未
完成的组合继续；未完成的单组会重新执行。每组三十三次查询写一次进度，
每完成一个组合写原子结果。进度和日志在 Git 忽略的
`data/stage_cvpr2027_experiments/external_policy_chain_v1/`。
先核对原进程是否仍在运行，不要因为日志暂时未更新就启动重复进程。

## 输入边界

外部接口 `ExternalPrefixAdapter` 只接受当前及之前的坐标，使用连续八个
原始标注步。它不自行打开数据集，也不授予数据使用角色。DUT 与 DroneCrowd
仍需要各自的数据准入规则，不能直接把保留集传进去试效果。

`scene_from_prefix` 构建 `SceneInput`；`FrozenView.infer` 返回候选预测、CV
基线、各策略的选择掩码和预测支持掩码。某个策略的最终坐标按以下方式组合：

```python
chosen = output["choices"]["half_joint"]
prediction = np.where(chosen[:, None, None], output["candidate"], output["baseline"])
valid = output["forecast_valid"]
```

必须保留 `valid`。历史不足的可见智能体仍在上下文里，其占位坐标不能被
当成有依据的静止预测。异常数值会回退；求解失败会标为未匹配，不会算作
联合优化成功。`half_joint` 是固定比较策略，不代表它已经获准部署。

## 如何理解结果

本轮在三十三段源录像中固定取首、中、末查询，共九十九个不同查询；十二
个模型/成本头/种子组合产生一千一百八十八次执行。这些重复执行不是独立
样本。全部重载复现通过，但联合选择与独立选择在这些查询中没有产生差异。
这不是交互贡献证据，也没有测量 FDE、ADE 或 easy degradation。

下一步是外部数据准入、有限场景数下的校准及一次性外部验证。若独立场景
支持不足，要明确报告不能认证风险，不能把录像或重叠窗口凑成独立场景。
主任务仍是八步观察、十二步预测；不同数据的原始步长不能直接换算成相同
秒数。当前不作 metric、true 3D 或 foundation 声明，Stage5C 与 SMC 关闭。
