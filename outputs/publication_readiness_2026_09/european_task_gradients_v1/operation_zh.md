# 冻结模型梯度诊断：复现与恢复

## 这轮实际做什么

读取上一轮已经训练好的 288 个成本模型，用每个模型原来保存的 256 条
训练批次计算梯度。分别克隆模型及 AdamW 状态，在内存中模拟一次纯成本
更新、一次成本加辅助分类更新，然后丢弃克隆。不继续训练或覆盖检查点。
最终检查的是损失干扰，不是新的预测准确率或部署收益。

`fresh_run`：梯度、虚拟优化器更新、汇总和复算。
`cached_verified`：源数据、因果预测、模型、训练划分及标签，均有哈希关联。
`not_run`：新的留出结果选择、独立风险校准、最终确认和部署。

## 本地运行

在仓库根目录使用原生 arm64 环境：

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_task_gradients.py --phase run
.venv-pytorch/bin/python scripts/run_m3w_european_task_gradients.py --phase report
.venv-pytorch/bin/python scripts/plot_m3w_european_task_gradients.py
.venv-pytorch/bin/python scripts/verify_m3w_european_task_gradients.py
```

诊断配置及源代码由注册锁绑定。注册提交必须先存在；不能改完代码仍冒用
原来的锁。独立复现需要本地原始来源和私有检查点，仅有公开汇总不够。
运行入口在导入 Torch 前拒绝 Rosetta/x86_64；CPU 4 线程、interop 1、
DataLoader workers 0，不探测或申请 GPU。

## 恢复与安全

心跳和事件日志位于
`data/stage_cvpr2027_experiments/european_task_gradients_v1/`。
意外中断后重复 `--phase run`，已完成的组会校验后复用；文件锁防止重复进程。
等待期间不因为暂时没有输出而杀掉进程。原始模型和优化器文件始终只读；
训练数据、检查点、PNG 预览不提交 GitHub。

验证器重新计算全部诊断、比对模型哈希、检查图像字节可复现，并运行相关测试。
历史同版本检查复用已有凭据，不将复用写成新跑；没有重跑整套历史测试。

## 解释边界

负梯度夹角并不自动等于 AdamW 让成本变差。虚拟更新只描述最终模型的一个
训练批次，不能证明完整训练路径的因果机制，也不能证明新方法会泛化。
后续真正修改训练方法必须另行注册、训练并评价，不能把这张图算作模型提升。
保持 8 步观察、12 步预测的主协议；不作米、秒、真实三维、基础模型或物理安全
声明。Stage5C 和 SMC 不执行。
