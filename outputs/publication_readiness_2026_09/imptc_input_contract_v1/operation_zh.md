# 本轮操作与续接

本轮不是重新训练或外部预测性能评估。实测的是：同一段过去轨迹换一个
坐标单位，已有预测器和风险头的输出是否保持一致。未来误差没有读出。

## 已完成

- 校验 2,937 个冻结来源绑定文件、24 个预测器和 36 个风险头。
- 在 IMPTC 样本的 58 个查询、754 个 past-only agent windows 上运行真实 Torch
  前向和固定风险头；只用预先指定的 seed17/coupa 排除模型。
- 确认两个原始单位特征足以改变大量风险头的净收益符号。这不是部署切换率。
- 新增独立无单位输入接口，未替换旧模型。旧 356 列风险头不能接新 355 列特征。
- 保留 EqMotion 的 3 个严格归一化容差失败；还原坐标后的差值较小，但不能
  因为换一种检查就把原先失败改为通过。
- 完整重放一致；64 个相关测试通过，旧全量测试未重跑。

## 可复现命令

在项目根目录运行，使用 arm64 环境、CPU4、interop1、workers0：

```bash
.venv-pytorch/bin/python scripts/audit_m3w_imptc_input_contract.py --verify
.venv-pytorch/bin/python scripts/check_m3w_imptc_unit_restore.py
.venv-pytorch/bin/python scripts/check_m3w_imptc_cost_units.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_unit_free_prefix.py tests/test_m3w_external_prefix_adapter.py tests/test_m3w_imptc_intake.py tests/test_m3w_native_forecast.py tests/test_m3w_native_eqmotion.py -q
```

第一条检查绑定文件和 checkpoint，再运行同版本数值前向、核对所有数组。
只在已有数据/代码哈希一致时复用；哈希不一致就停止，不覆盖旧证据。
原始数据、模型文件和数值探针在本地私有目录，Git 仅含代码、配置和轻量结果。
这些检查无需 CREATE；本轮没有远程提交作业，也没有改动其他项目进程。

## 下一步

1. 用单独版本修复 EqMotion 小量浮点扰动放大问题，仍不读取 IMPTC 未来误差。
2. 在已暴露的 SDD 训练来源上接入相同无单位特征，重新拟合 matched risk heads。
   保持 obs8/pred12、stride12 和所有数据用途边界。
3. 外部数据只有通过来源、观测模式、时间格和独立场景用途审查，才能进入
   正式校准/评价。IMPTC 已用于输入设计检查，不再称为完全未接触的确认输入。

常规审计、修复和复现由项目执行端承担，不要求用户手工审计。仍然禁止
Stage5C、SMC、未来输入、测试调参，以及未经验证的米/秒/真实3D声明。
