# 风险事件辅助误差学习：操作说明

## 目的与边界
检验风险事件辅助监督能否改善 expected H/H_E，而不是再次证明分类能学。
三组固定为 cost_only、cap_aux、shuffled_aux；只更改辅助监督。
上一轮分类门槛失败仍有效，不把分类结果或本轮训练完成写成部署成功。
主协议为观察8步、预测12步原生标注步长，检测器像素坐标。
独立选择、预留校准和确认数据不打开。Stage5C、SMC不执行。

## 环境与执行
使用原生 arm64 `.venv-pytorch/bin/python`，CPU4、interop1、workers0。
入口在导入 Torch 前拦截 Rosetta。先注册代码与方案并提交 Git，再运行：

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_cap_auxiliary_cost.py --phase support
```

支持检查提交后运行100次更新试跑；随后继续同一个检查点至2000次更新，
每200次保存模型、优化器、随机数、预处理和抽样记录：

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_cap_auxiliary_cost.py --phase pilot
.venv-pytorch/bin/python scripts/run_m3w_european_cap_auxiliary_cost.py --phase train --resume
```

全部预测冻结并提交后才能读出结果：

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_cap_auxiliary_cost.py --phase evaluate
.venv-pytorch/bin/python scripts/report_m3w_european_cap_auxiliary_cost.py
```

私有目录 `data/stage_cvpr2027_experiments/european_cap_auxiliary_cost_v1/`
保存 checkpoints、逐行预测、日志和心跳。心跳不是进程仍存活的证明，应
同时核对PID。正常慢速不重启；异常时保留上一个检查点再定位原因。
至少保留10GiB磁盘空间，不上传缓存、检查点、原始数据或逐行预测。

## 评价与复现
三个种子在每个场景内平均，再对四个场景做3000次配对 bootstrap。
六个组合存在重叠，滑动窗口不是独立样本。全量保留负结果和无法估计项。
费用覆盖比是预测/实际误差均值，不是 conformal coverage 或安全保证。
参考误差分量冻结，事件概率不乘入误差输出，zero-reference保护不变。

复现入口逐值核对全部检查点预测与全部结果，再重现图表和运行限定测试。
恢复检查点做推理不等于另一轮独立重训；合成单元测试不等于真实泛化。

在全部模型正常结束、prediction_freeze.json 已提交之后，依次执行：

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_cap_auxiliary_cost.py --phase evaluate
.venv-pytorch/bin/python scripts/report_m3w_european_cap_auxiliary_cost.py
.venv-pytorch/bin/python scripts/plot_m3w_european_cap_auxiliary_cost.py
```

图表检查及结论文档完成后运行最终复现：

```bash
.venv-pytorch/bin/python scripts/verify_m3w_european_cap_auxiliary_cost.py
```

该入口检查来源缓存哈希、全部432个检查点推理、144个留出结果、三组
抽样匹配、图表与报告字节一致性，以及8个限定测试文件。只有实际终止
成功且 verification.json 中 all_passed=true 才能声称这次复现通过。
这不是完整历史测试套件，也不是独立场景确认。

最终验证会绑定本目录报告和相关代码的哈希。若之后改动已绑定内容，
旧验证不再证明新版本；应记录新版本并重新核查，不覆盖旧证据记录。
不要删除检查点来强行重跑；已完成模型由记录校验，未完成模型从最新
完整检查点恢复。任何身份或输入哈希不一致都应先诊断。

## CREATE 与版本
本轮已只读查询CREATE队列，未提交、取消或修改远程作业。
训练放置依据本地真实试跑成本。若内存或运行成本不合适才转授权HPC。
代码、方案、轻量指标、报告和验证记录分阶段提交；不带入其他项目已暂存改动。
