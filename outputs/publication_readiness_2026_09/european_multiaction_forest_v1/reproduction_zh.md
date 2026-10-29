# 当前七候选强树模型对照：运行与核查

## 实验是什么

本实验训练的是连续代价估计器，不是新的轨迹预测网络。预测轨迹来自已冻结的
物理候选和神经预测器。六个上下文、三个控制器种子、三类候选库，共 54 个
ExtraTrees 模型，每个 128 棵树。控制器种子不等于三次独立训练的预测器。

输入为 398 个过去/静态场景/因果 rollout 特征，加七维动作身份和一个因果
envelope 特征。训练只用原拟合录像，重复原神经头的查询采样计数。推理不使用
未来端点、未来有效标签 mask、中心速度或测试端点目标。

树模型的对称五矩回归与神经头的非对称损失、约束解码不同。相同信息和采样，
不代表相同损失、计算量或只改变一个架构因素。旧二分类 forest 结果不能直接
当作本实验模型。

## 环境和数据

在 `/Users/yangyue/Downloads/World` 使用原生 arm64 的 `.venv-pytorch/bin/python`。
本轮 sklearn 为 1.8.0，CPU 计算线程 4，单进程，不启动 DataLoader 子进程或
joblib 进程池。不能用 Intel Conda/Rosetta 替代已核验环境。

公共仓库保存代码、固定配置、报告、轻量拟合收据和汇总指标；不保存原始轨迹、
图像、派生大矩阵或模型权重。实际训练/推理还需要合法取得的私有 SOURCE 数据、
父预测器和父 cost-head 检查点。仅克隆公共代码不等于具有全部复现资产。

## 首次运行顺序

下列是本轮已经采用的阶段命令。注册文件和完成产物是不可变记录，已完成阶段
不会被同名命令覆盖。新的独立重训应另建实验身份，保留此快照用于对照，不能
删除已有报告后将同一批留出数据重新称为首次测试。

```sh
.venv-pytorch/bin/python scripts/run_m3w_multiaction_forest.py register
.venv-pytorch/bin/python scripts/run_m3w_multiaction_forest.py pilot
.venv-pytorch/bin/python scripts/run_m3w_multiaction_forest.py train
```

所有 54 个最终模型及其收据提交冻结后，才运行：

```sh
.venv-pytorch/bin/python scripts/run_m3w_multiaction_forest.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_multiaction_forest.py verify
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_multiaction_forest_v1
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_multiaction_forest_v1 --verify
.venv-pytorch/bin/python scripts/report_m3w_multiaction_forest.py
```

`verify` 是重新推理核对，不是重新训练 54 个模型。全量试跑还实际比较了三库
8+8/直接16棵恢复，首个完整模型另做 128 棵独立重训，逐树核对。

## 正常等待和中断恢复

心跳和长日志保存在被忽略的
`data/stage_cvpr2027_experiments/european_multiaction_forest_v1/`。
每 16 棵树保存原子检查点。正式训练 PID 为 8500；PID 只对应这一次运行，
后续恢复需核对新日志，不能用旧 PID 断言进程仍在。

先确认原进程已结束，才可接续：

```sh
.venv-pytorch/bin/python scripts/run_m3w_multiaction_forest.py train --resume
```

不要并行启动重复作业；进程锁会拒绝第二个进程。输入哈希、采样、配置和
预处理改变时恢复必须失败，不能为了继续跑绕过检查。预留 10 GiB 磁盘，私有
模型总量上限 4 GiB。健康运行可以等待；试跑耗时投影不是完成证据。

## 如何解释结果

- `training_freeze.json`：54 个模型是否都完成、训练耗时/内存/模型大小。
- `verification.json`：144 组推理及指标是否精确重放。
- `summary.json`：全部候选库、固定敏感性、逐录像风险和配对区间，不只看最佳值。
- `readout.md`：已知标签收益、未知结果 completion 界、风险失败及范围说明。
- `gates.json`：科学 gate 与工程核验分开，跑完不等于可部署。

同切换数量对照只用于效用；空风险仍是不可估，不是安全的 0。完整策略仍须
满足所有要求录像的 all/easy 2% 正伤害要求。Hard/easy 补充指标按已知误差总量
聚合，不能代替主 locality 指标或完整风险；空 hard 录像不能被删除。

Completion 对比用固定已知 floor 误差总量归一化，不是对完全观测未来总体
误差下降百分比的识别。3,000 次 locality bootstrap 仍是暴露过的 SOURCE
开发区间，不是独立确认。

本轮数据是斜视 webcam detector-track 框中点像素，obs8/pred12/stride12
raw frames。不是 SDD 人工 gold、米、秒、true 3D 或 foundation。独立
selection/calibration/confirmation 和 DroneCrowd 未开放；Stage5C、SMC 不执行。
