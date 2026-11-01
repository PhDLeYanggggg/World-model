# 候选支持与联合预算验证复现

这轮实现了新的因果候选规则，但没有训练或部署新模型。只移除旧“预测收益必须为正”
门槛，保留可用动作、正预测参考量及两条2%成本预算。未来easy标签只用于诊断oracle。

```bash
.venv-pytorch/bin/python scripts/run_m3w_reference_support_preflight.py run
.venv-pytorch/bin/python scripts/run_m3w_reference_support_preflight.py verify
.venv-pytorch/bin/python scripts/run_m3w_query_easy_support.py run
.venv-pytorch/bin/python scripts/run_m3w_query_easy_support.py verify
.venv-pytorch/bin/python scripts/report_m3w_reference_support_preflight.py
.venv-pytorch/bin/python scripts/report_m3w_query_easy_support.py
```

当前完成目录不可覆盖；只有中断的未完成阶段可加`--resume`。原生arm64、CPU4、
interop1、workers0。两个完整阶段均保留72组/149,510目标/74录像。
前置完整验证89.699197秒，独立重算89.418772秒；联合必要支持检查82.377747秒，
独立重算84.496972秒。均为runtime_ready后计时，不含入口资产核验。

联合检查复用432个已验证pointwise证据，对其余12个视图重新检查。先用单agent或
非负成本符号条件判断，不能直接判断时使用二进制可行性问题，每query最多10秒。
求解器未最优或原始精确预算后检查不通过记“不可判定”，不算不可行或安全通过。
本轮6个阻塞视图有单agent直接证据，另3个包含数值求解器不可行判定，不能混称为
同样强的证明。必要easy支持仍不等于真实安全收益，更不是神经网络提升。

`registration.json`绑定代码、测试、协议与前序结果；`complete.json`绑定完整汇总；
`metric_groups.zip`仅包含标量。复算同时保持冻结预测矩和原始目标行哈希。
两个report命令只需公开汇总，不读取轨迹、特征、历史、latent cache或权重。
这不同于从原始数据开始完整匿名复现。相关检查：

```bash
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_reference_support_preflight.py tests/test_m3w_query_easy_support.py tests/test_m3w_reference_query_evidence.py
```

没有新增CREATE作业。本地完整运行峰值约11GB，已由实际运行核实；不是仅import测试。
以前SSH异常不能用来推断历史作业失败。完整仓库测试未由这些定向检查替代。
所有结论仍为EuropeanSquares暴露SOURCE、斜视检测器silver、本地图像坐标，观察8步、
预测12步、间隔12原始帧。不是SDD t50、米、秒、人工gold、true3D或foundation。
没有读取独立确认角色，没有改2%门槛，没有执行Stage5C或SMC。
