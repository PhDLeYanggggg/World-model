# 训练增益标签：运行与核验

这是补全监督标签，不是训练新模型。注册版本为 `f1b28fcf`；原预测器、数据角色、
预处理尺度、阈值和部署策略均未改变。不要将缓存内的未来标签接入推理输入。

## 已登记的运行顺序

工作目录 `/Users/yangyue/Downloads/World`，使用原生 arm64 环境。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/build_m3w_fitting_gain_labels.py pilot
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/build_m3w_fitting_gain_labels.py build --resume
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/build_m3w_fitting_gain_labels.py replay
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/report_m3w_fitting_gain_labels.py
```

仅在上一步成功且资源检查通过后进入下一步，不要无条件串联。首次试跑完成一组后，
用实际压缩大小的三倍估算全部108组存储，保留10GiB空闲。`build --resume` 保留已完成组；
锁文件用于互斥，不能单凭锁文件存在判断进程还在运行。以进程状态和心跳为准。

已完成的回执拒绝覆盖。重放重新读取原预测、原训练尺度及拟合标签，比较所有数组，
不按可能变化的压缩包时间戳判断一致性。不要修改旧缓存使重放“通过”。

每个组的监督标签只来自其两个 fitting 场景，不与 producer/controller/held 场景重叠。
原行ID、easy/reference/positive-harm标签必须与冻结checkpoint的身份哈希一致。
未知future标签保持NaN；不能把没有已知future的样本当成“零伤害”。

## 输出和边界

私有逐行结果在 `data/stage_cvpr2027_experiments/european_fitting_gain_labels_v1/labels/`，
心跳和过程日志在同级目录。不提交这些文件到GitHub。公开目录仅保留代码登记、
运行时间、汇总计数、验证回执和说明。

重复拟合行数不是独立样本数；unique row也可能是重叠窗口。新标签用于后续登记的
增益排序研究，本次没有评价新的utility模型，没有读取独立selection/calibration/
confirmation，没有改善部署gate的结论。继续保留pixel/raw-frame/silver的声明边界；
Stage5C和SMC关闭。
