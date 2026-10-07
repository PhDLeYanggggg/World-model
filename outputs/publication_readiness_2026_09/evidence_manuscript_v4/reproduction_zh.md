# 当前英文稿和实验复现

## 本版包含什么

本版把已完成的时间辅助监督实验及其训练集伤害诊断并入英文正文，不再把
216 个神经代价头写成尚未训练。六张表保留 SDD 早期研究、欧洲开发研究、
匹配神经实验及 TRAIN 回放的不同证据角色；两幅图只画公开汇总数值。
旧稿、旧失败和独立角色限制均保留。

文稿组装为 `fresh_run`，科学数值为 `cached_verified`。本版导出不是新训练、
新轨迹评估或新 bootstrap。新 easy-harm deviance 实验仍等待完整训练与
既定评价，不能提前写成有效方法。

## 本地轻量核验

在 World 项目根目录使用现有原生 arm64 环境：

```bash
.venv-pytorch/bin/python -m scripts.build_m3w_evidence_manuscript_v4 --check
.venv-pytorch/bin/python -m scripts.plot_m3w_evidence_manuscript_v4 --check
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_evidence_manuscript_v4.py
.venv-pytorch/bin/python -m scripts.verify_m3w_supplement_v4 --check
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_supplement_v4.py
```

导出器验证14个固定来源，以及链接的216个训练记录和72个评价分组记录。
这些是本地公开记录的哈希检查，不是重新读取远端所有私有 checkpoint。
正文修改放在 `manuscript.template.md`，再去掉 `--check` 重新生成。
检查不通过时核实变更来源，不能为了通过检查而随意更改来源哈希。

`supplement.md` 补齐了损失的实际权重、按 query 采样与平均、未知样本的
配对误差界、风险比例非单调反例，以及按 locality 聚合的统计步骤。
`supplement_examples.json` 是五类合成数学核验，不读取真实轨迹，也不更新
模型。不能把这些检查称为新增模型提升或完成匿名复现。

## 无需 CREATE 的汇总复现包

`aggregate_replay_v2` 可以在独立目录、仅有 Python 标准库的环境下复现六张
表和18条时间辅助对比指标。先在项目根目录生成本地压缩包：

```bash
.venv-pytorch/bin/python -m scripts.build_m3w_aggregate_replay_v2
.venv-pytorch/bin/python -m scripts.build_m3w_aggregate_replay_v2 --check
```

产物位于 `data/stage_cvpr2027_experiments/aggregate_replay_v2/review_bundle.zip`。
解压后运行 `python reproduce.py --check`，不需要仓库、远端账号、轨迹或权重。
这是14个公开汇总来源的哈希和表格核验；区间没有重新 bootstrap，原始标签、
训练及划分独立性也不由这个包证明。导出时核验的216个训练记录和72个评价
记录没有装入压缩包，因此不能声称在独立环境重验了这些记录。包内保留全部
负结果，并明确已去除常见身份标记不等于满足会议匿名要求。

## 真实训练和评价

已完成的时间辅助实验使用三个头种子、24个 TRAIN 上下文、三个辅助版本，
每次2000次固定更新；全部216个训练结果冻结后才评价七组模型。其结果不
支持推进。随后216个冻结模型的 TRAIN 回放不含新的优化器更新。

当前后继实验仅替换 easy-positive-harm 通道的一项损失。为处理历史数值
复现差异，登记了54个历史完全一致对照、18个原训练器重跑完全一致对照。
历史逐位复现仍不完整；不能把执行校验修复说成模型改进。

现有 CREATE 任务：训练数组 `37835856`，验收任务 `37835859`。
这是固定任务标识，不是要求重新提交。先读取状态：

```bash
.venv-pytorch/bin/python -m scripts.manage_m3w_easy_harm_deviance status --phase train
```

不要根据旧 heartbeat、连接超时或暂时没有队列输出判断任务结束，不要因此
重启训练。长任务按正常间隔观察；在确认终态后检查日志、退出码和产物。
必须保留已验收110个结果，只补缺失17个候选训练；另17个对照复用已验算
的 TRAIN 诊断状态。

全部144个最终结果及验收任务成功后，先收集并提交冻结文件，再运行：

```bash
.venv-pytorch/bin/python -m scripts.manage_m3w_easy_harm_deviance collect --phase train
.venv-pytorch/bin/python -m scripts.run_m3w_easy_harm_readout_v2 preflight
.venv-pytorch/bin/python -m scripts.run_m3w_easy_harm_readout_v2 run
```

冻结文件必须在受控 Git 提交后才能通过评价准入。评价若中断，只在核实同一
进程已结束后使用 `run --resume`；已完成分组逐字节验证，禁止换 checkpoint
或用开发结果调阈值。完整评价目录存在后不可覆盖重跑。

CREATE 数值工作只进调度器；固定四个计算线程、一个 interop 线程、零
DataLoader worker。保存优化器、随机状态、心跳和日志；不复用临床或
simulation 项目的数据/环境。保持10 GiB储存余量和256 MiB父子检查点上限。
本地只流式读取模型，不因空间不足下载 checkpoint 缓存，不删无关数据。

## 结果解释

- MSE更低不等于选择收益更高；时间辅助模型已经出现这一反例。
- 2%是选中样本的正 easy-harm/reference，不是净 easy ADE degradation。
- TRAIN回放只诊断拟合后的选择行为，不能称新场景泛化。
- 三个head seed不是三个独立forecaster；重叠窗口不是独立样本。
- 空分母与未知未来不能按零伤害处理。
- 当前仍未证明正向、独立、风险合格的方法，尚非投稿就绪。
- 不执行 Stage5C 或 SMC，不使用 metric、seconds、human-gold 或 true-3D 声明。
