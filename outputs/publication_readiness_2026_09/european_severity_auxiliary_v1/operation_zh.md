# 伤害加权辅助监督：运行与复现

## 这轮的改动

保持同一网络、初始化、采样、优化器和总步数，只把辅助分类损失按训练样本
的伤害程度加权。真实未来误差只作为训练标签/损失权重，推理仍只接收因果
特征、过去可计算的预测分歧范围和训练得到的归一化参数。

辅助输出现在对应伤害加权的 easy 比例，不是普通 easy 概率；不能把其分类
准确率当成校准成功，也不把该概率乘入预测成本。正式比较仍是风险成本误差，
原有分母、策略、门限和独立数据角色均保持冻结。

## 环境与运行

在仓库根目录使用原生 arm64 环境，CPU 4线程、interop1、workers0。
入口在导入 Torch 前阻止 Rosetta/x86_64。运行100步真实试跑后，从检查点
继续到每个模型2,000步。144个新模型、总288,000步，不额外添加试跑预算。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_severity_auxiliary.py --phase support
.venv-pytorch/bin/python scripts/run_m3w_european_severity_auxiliary.py --phase pilot
.venv-pytorch/bin/python scripts/run_m3w_european_severity_auxiliary.py --phase train --resume
```

注册及训练支持报告必须已经提交，才能训练。已有试跑不要重复执行：直接用
`--phase train --resume`。已有完整模型校验后复用，中断模型从最近200步检查点
恢复。心跳、事件日志、输入哈希和模型均在私有目录
`data/stage_cvpr2027_experiments/european_severity_auxiliary_v1/`。
锁文件仅防止并发；判断运行与否必须看真实进程，不凭旧心跳杀进程。

训练结束后先核对并提交预测冻结清单，再读出新的留出结果：

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_severity_auxiliary.py --phase evaluate
.venv-pytorch/bin/python scripts/report_m3w_european_severity_auxiliary.py
.venv-pytorch/bin/python scripts/plot_m3w_european_severity_auxiliary.py
.venv-pytorch/bin/python scripts/verify_m3w_european_severity_auxiliary.py
```

验证器检查逐行标签和预测对齐、初始成本与普通辅助模型一致、抽样及RNG一致、
缺标签从未抽取、训练权重只来自训练集、成本与排序指标独立复算，以及图表
字节级可复现。同版本历史测试复用凭据，未重跑时明确标记。

## 解释和权限边界

`fresh_run`用于本轮支持检查、新模型和新评价；已有来源及对照是
`cached_verified`。来源开发集早已暴露，不能重命名为独立最终测试。
窗口重叠，质量集中度ESS不是独立样本量。预测冻结不消除历史暴露。
训练loss降低不能代替留出成本gate，更不能代替世界模型贡献或可部署证据。

本地优先，若实测资源不够才按已批准CREATE限制转移；此次CREATE查询只读，
不修改其它任务。保留10GiB磁盘空间，不上传数据、缓存、权重或第三方素材。
8步观察/12步预测仍是annotation steps；不作米、秒、人工gold、真实三维、
foundation或物理安全声明。Stage5C/SMC关闭，正式论文提交仍由用户确认。

## 本轮结果怎么解读

144个模型、288,000步已经完成，不是训练计划。预测冻结提交`41a478ca`已在
新结果读出之前推送。伤害加权辅助监督没有通过原定成本gate：相对原始模型，
完整特征6组比较中0组区间为正、1组为负、5组跨零；点估计范围为-20.14%到
+2.38%。跨零不代表已经证明等效或安全。尾部识别和伤害总量校准也有负结果。

这里比较的是easy样本伤害的预测误差，不是轨迹FDE提升。loss下降、事件排序
部分改善，都不能代替成本估计准确或部署成功。保留原有冻结策略，不切换到
这批新模型。下一步先定位录像/轨迹层面的误差与训练权重集中情况，再决定一项
有依据的修复，不依据本轮留出分数连续搜索门限。完整数字见`results.md`，
数值复算与测试记录见`verification.json`；验证通过也不改变科研gate失败。
