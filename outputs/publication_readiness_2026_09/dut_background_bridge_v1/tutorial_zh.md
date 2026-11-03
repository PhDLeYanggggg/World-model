# DUT 来源比较与新开发数据入口

## 本轮实际结果

已从作者固定版本取得全部 28 张背景合成图，成功图像载荷 19,357,004 字节。
没有下载视频，也没有执行作者程序。完整比较包括 Moving-to-DUT 3,360 对、
DroneCrowd-to-DUT 3,136 对、DUT 内部 378 对，总计 6,874 对、19,866 次
图像比较。匹配约 459.18 秒，峰值内存约 659 MB，适合本机，未新增 HPC 作业。

两个跨来源方向都没有支持候选；DUT 内部有 171 对保守关联，207 对不支持。
没有发现连接 DUT/DroneCrowd 保留角色的组件；全部既有分组和 DUT 两个
作者地点组保留。这里的“没有匹配”不是地理独立性证明。

正式复核约 11.25 秒：28 张原始 PNG 和预处理特征重提一致，696 个父缓存
视图及 6,874 对记录/分组一致，再重算固定 60 对，一致。额外只读复核约
9.45 秒再次通过；不是第二次全量匹配或再次下载。

三页 DUT 预览和八张内部候选面板均已检查，准确范围在 visual_review.md。
这些是 self-audit，不是 human gold。背景合成时间未知，且部分裁剪图存在
最高约 1.74 的横纵缩放比；局限不隐藏。不能复制一张背景图假装多帧，也
不能把合成背景放进 past-only 神经网络输入。

## 已开放的开发数据

通过另一个仅拟合用途的接纳记录，64 个 MovingDroneCrowd++ 发布方训练
片段现可用于开发拟合，共 136,295 个完整原生 8/12 窗口、24 个保守分组。
分组不是已验证独立物理地点；窗口不是独立样本。其余片段不进入本轨道
预测用途，DUT/DroneCrowd 保留角色不变。

接纳文件位于 `outputs/publication_readiness_2026_09/moving_development_admission_v1/`。
这是数据入口，不是训练完成、模型升级或完整 producer-chain 通过。实际
模型、teacher、预处理和 critic 的来源排除仍须在下一实验冻结并执行。

## 可重复核对

在 `/Users/yangyue/Downloads/World` 使用原生 arm64 环境：

```sh
.venv-pytorch/bin/python scripts/check_m3w_dut_background_bridge.py
.venv-pytorch/bin/python scripts/report_m3w_dut_background_bridge.py
.venv-pytorch/bin/python scripts/admit_m3w_moving_development.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_dut_background_bridge.py tests/test_m3w_dut_background_summary.py tests/test_m3w_moving_development_admission.py
```

第一条只读复核全部保存结果、来源和固定小前缀，不下载、不写应用产物。
第二条重建标量报告，不训练。第三条在同一文件版本下重新核验接纳记录
和全部 64 份原标注身份；已完成记录只允许内容一致，不会新发模型结果。
复现需要私有源数据和缓存，GitHub 不包含这些媒体或数组。

已完成的 acquire/pilot/run/verify 阶段不可覆盖。若未来有真正未完成任务，
先确认旧 PID 已退出，再对同一配置使用 `--resume`；不能因工具观察超时
重复启动。本轮全部任务已结束，不需要重新匹配。

当前没有新增 FDE/ADE、loss、risk 校准或部署提升。下一步是使用新拟合
数据检验因果历史/共享场景信息的真实预测贡献。坐标仍是像素/数据集本地
单位，原生 annotation step 未验证为秒。Stage5C 和 SMC 均不执行。
