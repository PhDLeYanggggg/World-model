# 当前候选库成本模型：运行、恢复与结果核查

## 实验目的和当前状态

这轮对照固定现有轨迹预测器、输入、划分、风险标准和阈值，只检验更强的
常规非线性成本估计器能否改善选择。不是新训练JEPA、Transformer或轨迹
动力学模型。两种ExtraTrees分别使用单行特征和同一时刻其他智能体的因果
特征摘要，9个生产者管线各训练两种，共18个128树模型。

2026-11-02 14:49 UTC，正式18个模型已完成，共2304棵主要树。训练结束
不等于主要科学结论通过。最终结论以完整`summary.json`和独立
`verification.json`为准；这两个文件尚未生成时，开发集主要结论就是not_run。

## 环境

从项目根目录使用原生arm64的`.venv-pytorch/bin/python`，不用x86 Conda。
本实验固定拟合线程4、确定性预测线程1、Torch交互线程1、workers0；不要
中途把线程设置、树数、深度或输入改掉后继续写入原实验ID。

过去的Intel OpenMP/SHM问题不能凭一次等待就推断复发。本次中断是原进程
在轮次中断后已不存在，退出码和退出机制未知，没有观测到OpenMP卡死。
健康进程有持续CPU消耗且检查点推进；慢不等于卡死。当前实验的12小时
边界、10GiB磁盘储备和原始训练预算保持不变。

## 数据和结果来源

- cached_verified：旧预测器、7条候选轨迹、444维因果输入、SOURCE划分、
  标准化参数、标签和原神经控制器完整采样访问次数。
- fresh_run：18个成本森林的训练、后续冻结推理、开发集评价和独立复核。
  只有对应收据实际生成且校验通过，才能标记完成。
- not_run：没有执行的阶段；不能用注册文件、旧控制器或试跑替代。

每个成本模型只使用其轨迹生产者未训练的8个SOURCE场景。未来标签只进入
训练损失和评价，不能进入因果输入。未知标签没有被填成零误差。6个模型
选择场景已经暴露，只能叫开发集，不能通过改名恢复为独立测试。
本轮不读取私有校准、最终确认或DroneCrowd数组，也没有新提交CREATE任务。

## 阶段入口

以下是按阶段调用的入口，不是在已完成实例中无条件重复执行的脚本。
每个已完成阶段不可覆盖。先查看已有收据；未完成且检查点仍绑定原配置、
数据和代码时，才使用`--resume`。不要删除收据或挪走旧结果伪装fresh。

```sh
.venv-pytorch/bin/python scripts/run_m3w_selection_cost_forest.py pilot
.venv-pytorch/bin/python scripts/run_m3w_selection_cost_forest.py train --resume
.venv-pytorch/bin/python scripts/diagnose_m3w_selection_cost_forest_fit.py
.venv-pytorch/bin/python scripts/run_m3w_selection_cost_forest.py infer --resume
.venv-pytorch/bin/python scripts/run_m3w_selection_cost_forest.py eval --resume
.venv-pytorch/bin/python scripts/run_m3w_selection_cost_forest.py verify
.venv-pytorch/bin/python -m scripts.report_m3w_selection_cost_forest
.venv-pytorch/bin/python scripts/plot_m3w_selection_cost_forest.py
```

`report`必须用`-m`入口，且要求完整18个模型和54组指标已独立核验。
所有18个开发集推理和动作先冻结，再开始本轮标签评分。不能先看有利种子
或场景再替换主要对照。主要指标固定为context_forest相对mean_absolute的
完整直接配对completion-lower差，不是挑选更有利的row或context结果。

公开代码仍依赖未随Git发布的SOURCE数组、旧模型和原有角色资源。因此
当前仓库不是从公开原始数据一条命令冷启动的完整匿名复现包；已有资源上的
精确复核也不能冒充独立科学确认。这是投稿材料仍需补齐的真实缺口。

## 本次实际恢复

原正式进程PID38473停止后，9个完整模型及第10个模型96/128树状态保留。
恢复前复制了96树检查点，SHA256为：

```text
e5a18ff49128a41aaeba6583b41e9118be4695d5ac24506db38c418bc7e33337
```

原实例由PID50966继续，完整旧模型先验证，不重新拟合；第10个模型已经
补齐至128树。`turn_interrupt_recovery.json`记录当时检查点、输入哈希和日志。
它是历史事件收据，不是当前实时进度。原中断进程内未保存树块的实际工作量
未知，不能算进已保留的正式树数；恢复进程耗时也不是整个训练总耗时。

独立复现进程PID54254从头训练真正被中断的生产者，用完整128树和恢复后
结果对比，包括前96树和损失监测轨迹。它有独立检查点、锁和heartbeat，
不会覆盖主训练的状态。完成证据是`actual_interruption_refit.json`，在此
文件生成前不能写“实际中断已精确复现”。本次检查已经完成，完整128树
及保留96树前缀/轨迹均精确一致，耗时1152.128秒，额外128棵复现树独立
计数。首个生产者的完整重训也已精确一致，但与第10个模型是两项不同检查。

## 只读观察

```sh
ps -p 50966,54254 -o pid,etime,time,%cpu,rss,command
tail -n 8 data/stage_cvpr2027_experiments/european_selection_cost_forest_v1/train_resume_after_turn_interrupt_20261102T134823Z.log
tail -n 8 data/stage_cvpr2027_experiments/european_selection_cost_forest_v1/actual_interruption_direct_refit.log
df -h .
```

PID是本轮记录，之后可能退出或被系统复用。观察超时、连接失败和没有新
日志，单独都不足以证明训练卡死或需要重启。先核对进程、检查点和阶段
收据；不要重复启动主训练。如果保护触发，保留已有状态，不删除数据、
缓存、系统文件，不降低预算或擅自放宽保护阈值。

## 补充诊断不替代主要实验

SOURCE候选库支持诊断在444个重复录像视图中都找到满足原要求的动作，
但用了结果标签，只能说明存在可行候选，不说明模型能识别它。
另一个零更新几何投影只把SOURCE拟合目标平均降低约0.37%，没有训练
新权重，没有评价或升级部署策略。两项结果都不能替代森林完整主要对照。

损失监测值是固定SOURCE子集上的未投影拟合目标，不是完整验证误差。
`decoded_SOURCE_objective.json`才使用原查询访问次数计算完整解码目标；
它仍是拟合重代入，不是泛化。不同损失定义、RMS或加权方式不能混排大小。

## 科学结论边界和Git

统计区间使用6个开发场景的3000次配对重采样，先平均生产者和实际训练
种子。252个重复录像视图不是252个独立场景。每个必需完整策略视图都要
满足原2%正伤害、easy净退化和有限参考支持；空分母不是安全零。均值有利
或违规减少不能覆盖主要指标、安全条件或独立确认失败。

观测8步、预测12步，步距为12个原始帧，单位为本地图像像素。可见标签ADE
不是完整12步ADE，整行未知的界也不补齐部分未来中的缺失步。silver不是
人工gold，不声明米、秒、物理安全、true 3D或foundation。Stage5C和SMC关闭。

Git只提交代码、配置、说明和轻量标量/自有SVG图。数据、模型、缓存、
检查点、视频图像、第三方文件及`.venv-pytorch`保持私有。作者账号资料也
不进入匿名包。本轮每个有意义结果均更新结果页和研究状态并安全同步。
