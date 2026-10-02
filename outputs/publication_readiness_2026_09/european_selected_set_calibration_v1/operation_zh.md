# 源域选择集校准：运行与核验

本页对应 `european_selected_set_calibration_v1`，不是早期 Stage37 部署教程，
也不是完整投稿实验已经完成的声明。旧操作记录中的“CREATE 仅只读、目录未确认”
描述的是当时状态；本实验使用已经交接并授权的 M3W 独立运行环境。

## 本轮实际做了什么

复用并校验 72 个冻结预测头及源域输入，只重新拟合经验校准系数，不训练新的
Transformer、JEPA 或轨迹网络。新增对照按实际保留的样本重复计算校准残差，
整段留出录像验证。所有阈值和迭代上限在运行前固定，不使用迁移测试结果选参数。

观察 8 步、预测 12 步、raw stride 12、image-local detector-silver 协议不变。
2% 指选中样本的正伤害与参考误差之比，不是随意替换为全体 easy ADE 的 2%。
未知标签仍是未知；没有分母或没有介入，不能算风险通过。

## 查看状态，不重复提交

在项目根目录使用原生 arm64 环境：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/manage_m3w_selected_set_calibration.py inspect
```

对应 CREATE 作业为 `37714473`，现已返回 `COMPLETED|0:0`，调度器用时 2 分 10 秒。
72 组新校准均完成，并完成相同输出的第二遍重放。不要再次运行 `submit`。
`inspect` 会读取已授权目录、作业记录和产物哈希，不训练模型，不修改 simulation。
排队、连接超时、读日志失败不等于训练失败；先核查原作业，不新建替代作业。

## 独立算术检查

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/verify_m3w_selected_set_readout.py verify
```

这会从 CREATE 读取已经校验的输入与输出到内存，独立重建因果决策，逐项核对
伤害、参考误差、未知标签上界以及场景级 bootstrap。它不是训练；CPU 持续计算时
可以等待，不按固定几分钟的时限杀掉进程。原始数组不写入本地缓存，只有轻量汇总
和核验凭据写入报告目录。本机数值缓存仍保留原定 10 GiB 空间要求。

核验返回成功前不能把结果写成“已独立核验”。同一执行者的另一套算术实现也不等于
另一研究团队的独立复现。调度器完成、代码测试通过、算术一致和方法假设成立是
四件不同的事。

本次 v2 已实际核验成功：19,872 项标量、864 个决策哈希和 168 项汇总/置信区间
检查一致。首次核验器的逐行重复解压问题已单独修复并登记，不是修改实验来获得
有利结果。主对照完整通过数仍为 23/72，效用下降；详见
[结论](conclusions.md)与[失败分类](failure_analysis.md)。没有升级模型部署。

针对性工程检查：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m pytest -p no:cacheprovider tests/test_m3w_selected_set_readout.py tests/test_m3w_selected_set_calibration.py tests/test_m3w_selected_set_runner.py tests/test_m3w_component_calibration.py tests/test_m3w_unknown_outcome_bounds.py tests/test_m3w_selected_pool_accounting.py -q
```

这不是全仓库历史测试通过的声明。相同版本已经通过的检查不必反复重跑；代码、
输入哈希或结果出现变化时，应补相应检查，而不是修改旧登记使其“通过”。

## 恢复与证据边界

运行器支持相同代码、配置和输入下的 `--resume`，但本次作业已经完整结束，不应
使用恢复入口。未来若确实中断，先确认作业终止并校验已写入的每组产物，再记录
恢复提交；不得覆盖不同结果。训练和较重科学计算通过调度器，不在登录节点执行。

当前结果只能解释已暴露开发数据中的源域校准。独立选择、校准、最终确认数据
没有开放；没有新迁移评估，没有部署升级。没有 metric、seconds、human-gold、
真实 3D、foundation 或物理安全主张，Stage5C 与 SMC 均未执行。
