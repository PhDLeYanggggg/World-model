# 新数据接入与复核

## 本轮完成了什么

本轮只完成 MovingDroneCrowd++ 的版本固定、稀疏下载、逐行几何转换、
历史/未来分离检查和发布方划分解释。没有训练模型，也没有读取预测误差。
现有 DroneCrowd 的112段整源确认保留规则没有改变。

观察8步、预测12步的“步”是本数据的原生标注索引，不是秒，也不能直接
与其他数据的 raw-frame t+50 混合。全源存在245,018个完整窗口，不代表
245,018个独立样本；44个发布方场景标签也不是已认证的独立地点。

## 环境

在项目根目录使用 `.venv-pytorch/bin/python`。本轮只有NumPy、requests、
标准库和现有读取内核，不导入Torch，不安装或运行作者的训练代码。
不使用默认x86Conda，不改动CREATE或simulation项目环境。

## 空目录首次执行

先核对已提交的配置、协议、源版本和整源保留规则。脚本要求实现与Git
中的冻结版本一致；不得为让命令运行而删除已有结果或解除保留限制。

```sh
.venv-pytorch/bin/python scripts/intake_m3w_movingdronecrowd.py register
.venv-pytorch/bin/python scripts/intake_m3w_movingdronecrowd.py acquire
.venv-pytorch/bin/python scripts/intake_m3w_movingdronecrowd.py verify
.venv-pytorch/bin/python scripts/review_m3w_movingdronecrowd_splits.py
.venv-pytorch/bin/python scripts/review_m3w_movingdronecrowd_splits.py --verify
```

这些阶段使用不可覆盖的完成记录。当前机器已完成以上步骤，再次执行
同一已完成阶段会拒绝覆盖，这是预期行为，不是重新计算成功。
下载若在完成前中断，可使用 `acquire --resume`，复核已保存成员后续读。
这不会重新划分数据、扩大下载或降低磁盘余量。

## 当前工作区可反复运行的只读复核

```sh
.venv-pytorch/bin/python scripts/check_m3w_movingdronecrowd_receipts.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_movingdronecrowd_intake.py tests/test_m3w_movingdronecrowd_splits.py
```

只读复核会重新检查缓存SHA和保留规则，重新解析全部120段并计算结构性
窗口统计、前缀不变性和划分统计；不写文件、不发网络请求、不运行预测。
它是cached_verified复核，不是再次下载或新的独立科学验证。需要当前机器
忽略目录中的源标注；仅下载Git里的汇总报告不能复现这些读取检查。

## 结果在哪里

- `analysis.json`：首次固定规则的全源结构统计，保留scene-only标识未解释状态。
- `split_interpretation.json`：按固定作者读取代码解释场景/视频混合清单。
- `execution.json`、`verification.json`：首次下载/处理与单独缓存复核。
- `split_execution.json`、`split_verification.json`：划分解释与重复复核。
- `source_manifest.json`：版本、成员CRC/SHA、传输量；没有整包本地SHA认证。
- `conclusions.md`：数量、版本差异、val/test场景共用及下一步。

所有原始标注、图像、权重和缓存均不提交Git。下一步是稀疏图像的源间
重叠检查和正式数据角色接入，不能把“读通了数据”写成“模型已经提升”。
