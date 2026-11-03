# 本地运行与结果核对

这轮不是 small demo：九个真实 Torch 模型均跑满 4,000 次更新，另有完整首模型
重训及真实断点恢复试跑。主要结果是 SOURCE 上的观测标签预测改善，不是新的
安全部署通过，也不是独立最终测试。主比较与控制器白名单比较不能混在一起。

## 环境

使用原生 arm64 `.venv-pytorch/bin/python`，不要使用 Rosetta 下的 x86 Conda。
固定 CPU 计算线程 4、inter-op 线程 1、DataLoader workers 0。不启用资源探测或
多进程加载。实际训练、重训和完整推理复算已运行；MPS 对等测试仍是可选跳过项。

训练数据、官方第三方代码、权重和预测数组均保留在本机忽略目录。公开文件只有
代码、配置、协议、轻量指标和自己的图表。运行前会检查源码、数据角色和输入哈希，
不要修改已冻结定义后继续使用原来的训练身份。

## 已完成实验的核对

以下两个命令重新汇总公开标量，并检查它们与完整复算凭据一致，不会重新训练：

```sh
.venv-pytorch/bin/python scripts/report_m3w_european_eqmotion.py
.venv-pytorch/bin/python scripts/report_m3w_european_eqmotion_controls.py
```

`summary.json` 和 `strong_controls/summary.json` 分别保存主比较与扩展控制。
`verification.json` 记录 72 组主指标复算，扩展目录记录 144 组复算。
`inference_verification.json` 记录完整模型推理重跑，不只是读取旧分数。
这不是新的独立数据确认。完整匿名冷启动复现仍需要补齐上游数据与环境交付。

## 中断与恢复

未完成阶段使用对应命令恢复；已完成阶段的凭据不可覆盖，不要删除旧文件冒充新运行。
以下是本轮实际执行过的阶段入口，不是让已经完成的目录重新从头跑：

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_eqmotion.py pilot --resume
.venv-pytorch/bin/python scripts/run_m3w_european_eqmotion.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_european_eqmotion.py infer --resume
.venv-pytorch/bin/python scripts/run_m3w_european_eqmotion_controls.py infer --resume
.venv-pytorch/bin/python scripts/run_m3w_european_eqmotion.py eval
.venv-pytorch/bin/python scripts/run_m3w_european_eqmotion_controls.py eval
.venv-pytorch/bin/python scripts/run_m3w_european_eqmotion.py verify
.venv-pytorch/bin/python scripts/run_m3w_european_eqmotion_controls.py verify
```

完整推理/指标验证阶段也会封存自己的运行凭据；不能把同目录重复覆盖凭据当作
新的可复现证据。当前可重复运行的公开标量报告与本轮实际完成的完整推理证明要分开。
需要另一次独立完整重放时，应先增加保留旧凭据的新重放目录/入口；这里不宣称该
匿名冷启动或多次重放交付已经完成。

本机权重在 `data/stage_cvpr2027_experiments/european_eqmotion_v1/trials/`。
训练日志、心跳、采样器、优化器和随机状态都在同一忽略目录下。首次完整重训比较
的是科学状态，不是带不同计时/序列化信息的 checkpoint 文件字节完全相同。

## 如何读分数

原始 SOURCE 人群有 318,969 个 target windows，白名单只有 149,510 个。
百分比差是使用共同 CV 分母的百分点，不是对每个 PCT 的相对误差下降百分比。
轨迹窗口会重叠，不能当作独立人数。Source 交叉拟合也不能写成独立最终测试。

图左边是已观察标签上的改善，右边是完全缺失未来行的保守下界；右边宽区间不等于
已观察到灾难性误差。部分缺失轨迹内部的步骤也没有被这些界填补。原始每录像
2% 正伤害、easy-net 和有限支持规则没有放宽；目前没有新策略通过部署检查。

当前是像素/raw-frame/检测跟踪 silver，不是米、秒、人工 gold、true 3D 或
foundation world model。Stage5C 不执行，SMC 不启用。
