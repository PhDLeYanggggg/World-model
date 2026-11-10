# M3W v202 复现顺序

先分清三件事：重放论文数字、重做有原始数据的评估、从头训练。匿名包支持第一件；
后两件需要原始数据及本地缓存或已有 CREATE 产物。不要为了复核表格重复提交训练。

## 本机环境与断点

在项目根目录确认 `uname -m` 和 `.venv-pytorch/bin/python` 都是原生 arm64。
不要从 x86_64 Conda/Rosetta 启动训练。计算线程和读取进程分开：OpenDD 用
四个 Torch 线程、零 DataLoader worker。下面的 CPU 检查只重建被遮盖的过去位置，
不会打开预测标签；需要已转换的 `ucy_students03`。`RUN_ID` 应使用新的名字。

```sh
RUN_ID=local_cpu_check
.venv-pytorch/bin/python scripts/probe_m3w_training_runtime.py \
  --run-id "$RUN_ID" --device cpu --threads 4 --rows 512 --steps 12 --checkpoint-every 6
.venv-pytorch/bin/python scripts/probe_m3w_training_runtime.py \
  --run-id "$RUN_ID" --device cpu --threads 4 --rows 512 --steps 24 --checkpoint-every 6 --resume
```

检查 `data/stage_cvpr2027_causal/runtime/$RUN_ID/latest.pt` 和运行报告，确认
第二次从第 12 步继续并写出第 24 步，而不是重跑。更换设备或数据须新建运行 ID。
本轮实测四线程、零 worker，从 12 续跑到 24，检查点加载后的对照更新参数最大差为 0；
这是短时运行检查，不是模型性能结果。
若单独测试 MPS，把 `--device` 设为 `mps` 并用新的 ID；设备失败不计为 MPS 成功，
本稿两组 OpenDD 公开架构对照在 CPU 上完成。慢时看心跳和步数是否持续变化。

## 本轮论文与实验

- [OpenDD 公开架构](../opendd_vehicle_stgcnn_v1/README.md)：已训练权重先核对
  `freeze.json` 和数据身份。`fit` 可沿同一权重、优化器和随机状态继续；不要改 held
  location 或 seed。`score` 只接受完整冻结清单。
- [Songdo](../songdo_failure_cases_v1/README.md)：用已经评分的 160 趟复核三例。
  先核对校准和确认范围，不能把补充案例当作新测试，也不重新调阈值。
- 解开 `review_supplement_v34.zip` 后在其目录执行 `python verify.py`。
  它复核匹配的标量统计和案例图哈希；不会重训。主稿依赖 `pypdf` 和已缓存的
  CVPR 样式与 Tectonic 资源。`TECTONIC` 指向本机 Tectonic 可执行文件：

```sh
.venv-pytorch/bin/python scripts/build_m3w_focused_evidence_v202.py --compiler "$TECTONIC"
```

## CREATE

先在已授权的 CREATE 会话用 `sacct` 查看已有编号和退出码；例如本轮核对的
`37835856_0`、`37835859`、`37837636`、`38287008`。结束后核对产物、配置哈希、
日志和正式读出；任务完成不等于模型通过。对应冻结清单远端与本地 SHA-256 一致，
原科研读出仍是未达标的开发结果。本轮没有提交新作业或下载权重。

需要新训练时按[已执行过的提交与恢复流程](../european_temporal_auxiliary_v1/create_reproduction_zh.md)
在调度节点申请资源，不在登录节点训练。保留作业号、提交回执、环境、日志和心跳。
重连先查原 job ID；查询失败不是作业结束。恢复使用同一数据和配置哈希、已核对的
最后一份完整 checkpoint。跑完后独立检查结果数值、原定划分和源级不确定性。
