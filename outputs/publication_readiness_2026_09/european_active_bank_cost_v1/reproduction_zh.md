# 匹配候选库成本学习：复现与结果解读

## 这轮做了什么

这不是继续跑旧 SDD t50，也不是只读取已有报告。本轮在原有6个拟合上下文、
3个成本头种子和3套候选库上，真实训练54个 PyTorch 成本头。每个头完整训练
2,000步，合计108,000主要更新，另有2,000步完整独立复训和160步真实试跑。
轨迹预测器本身是上一轮冻结的已训练模型；不能把54个成本头说成54个新轨迹模型。

原有六个物理候选、回退策略、数据行、采样、归一化、风险规则保持不变。
只替换第七个神经轨迹候选，同时重建它的因果特征、场景字段、分歧上界和监督代价。
新旧第七个候选虽然编号相同，但轨迹不同，必须分别使用各自真实代价。

## 环境与运行成本

在项目根目录执行，使用原生 arm64 环境，不能使用 x86_64 Conda/Rosetta。

```bash
.venv-pytorch/bin/python -c "import platform; print(platform.machine())"
```

预期为 arm64。运行器在导入 Torch 前阻止错误架构。CPU计算线程4、interop1，
DataLoader workers0，不做历史问题中的资源探测。CPU多线程与 DataLoader
多进程不是一回事，单进程加载不等于只用一个计算线程。

已完成的本机实测：缓存149.43秒/峰值2.20GB，真实试跑13.77秒/9.75GB，
54个完整训练282.96秒/11.01GB，首次完整策略读出126.49秒/11.07GB。
因此本轮本地资源足够，没有降级实验预算，也没有新提交或取消 CREATE 作业。

## 文件与来源

- 公开配置：`configs/m3w_active_bank_cost_v1.json`。
- 注册、完整训练与指标：本目录中的 JSON、报告、真实曲线和轻量聚合压缩包。
- 本地大数组、权重、日志：`data/stage_cvpr2027_experiments/european_active_bank_cost_v1/`。
- `fresh_run`：本轮新建候选缓存、真实训练、首次策略评估、另进程完整推理重跑。
- `cached_verified`：旧数据、分组、已训练轨迹模型、原有控制头和冻结缓存，经哈希核验。
- `not_run`：控制器持出的89个录像结果、独立选择/校准/确认、DroneCrowd结果、联合场景决策。

大数组和权重不进入 Git。只下载公开仓库不能从零重现全部实验；需要已有授权的
原始数据、之前的检查点和本地缓存。缺资产应明确报缺，不能伪造完成。

## 执行顺序和恢复

下列为实际入口，不是另一个缩短版 demo。已完成实验保留原始身份，不应覆盖重跑。

```bash
.venv-pytorch/bin/python scripts/run_m3w_active_bank_cost.py register
.venv-pytorch/bin/python scripts/run_m3w_active_bank_cost.py bank --resume
.venv-pytorch/bin/python scripts/run_m3w_active_bank_cost.py pilot --resume
.venv-pytorch/bin/python scripts/run_m3w_active_bank_cost.py train --resume
.venv-pytorch/bin/python scripts/evaluate_m3w_active_bank_cost.py register
.venv-pytorch/bin/python scripts/evaluate_m3w_active_bank_cost.py run --resume
.venv-pytorch/bin/python scripts/evaluate_m3w_active_bank_cost.py verify
.venv-pytorch/bin/python scripts/report_m3w_active_bank_cost.py report
.venv-pytorch/bin/python scripts/report_m3w_active_bank_cost.py figures
.venv-pytorch/bin/python scripts/report_m3w_active_bank_cost.py bundle
```

注册、代码和相应冻结收据必须先提交，运行器会核对。完整训练每100步存检查点、
每50步写心跳，记录 PID、时间和内存；12小时硬边界、10GiB剩余磁盘保护。
`--resume` 核验并跳过已完成头，不会把跳过说成新训练。进程锁防止重叠执行。
完整 `run` 已完成后会拒绝覆盖，此时使用 `verify`。缺少聚合 JSON 而已有
公开轻量包时，可以先恢复原字节；这不是新的科研重算：

```bash
.venv-pytorch/bin/python -c "from pathlib import Path; from src.evaluation.m3w_metric_bundle import verify; print(verify(Path('.'), Path('outputs/publication_readiness_2026_09/european_active_bank_cost_v1'), restore=True))"
```

不要在还有正常心跳的情况下另起重复训练。观察失败不等于训练卡死，应先读
已有日志、检查原 PID 和锁，再决定是否需要恢复。

## 指标不是同一个意思

已知标签 ADE 收益：物理1.946%、旧神经1.969%、新神经1.893%。新减旧为
-0.075个百分点，区间跨零，因此本轮没有证明替换候选能让完整策略更好。
同一个新成本头禁用神经选项后，已知标签收益下降0.166个百分点，说明实际使用
神经轨迹有小的局部收益；但它的缺失结果保守区间跨零，也不能宣布安全。

最重要的失败是完整策略只有37/444个必需录像风险视图通过，313个已知 easy
正伤害风险超标。总体 easy 平均收益为正，不能抵销受损 easy 样本的风险。
这里的2%规则没有放宽；同切换数量只是效用对照，空对照明确“不可估”。

149,510个去重查询身份被重复放在上下文/种子中，形成897,060行视图，
不能把后者当独立样本。区间是3,000次 locality 重采样，先平均上下文和种子。
SOURCE 历史上用于开发；即使逐字节复现，也不会变成独立测试。

## Loss 与限制

图里的 loss 是实际训练日志：连续五矩回归、低估伤害加权和行内排序损失，
不是分类准确率，也不是独立验证集 loss。降低 loss 不代表完整风险规则通过。
所有54个头完整曲线公开保存，缺标签行没有观测代价损失。

本轮为 detector-silver 包围框中点 pixel，观察8步预测12步，间隔12 raw frames。
不是历史 SDD t50，不是米、秒、人工 gold、已验证实时传感输入、true3D或
foundation 证据。Stage5C未执行，SMC未启用。下一步是修复代价估计与风险决策
的错配，而不是把这一轮失败更名为成功或重复用测试结果调阈值。
