# M3W 论文：本机与 CREATE 复现入口

当前候选：[v179 主文构建说明](../outputs/publication_readiness_2026_09/evidence_manuscript_v179_exact_exchange/README.md)。
这份短教程区分匿名汇总复算、真实检查点复跑和重新训练。三者的证据范围不同。

## 1. 匿名补充包：只复算已冻结结果

本机补充包为 `data/stage_cvpr2027_experiments/submission_admin/review_supplement_v8.zip`。
解压到空目录，从该目录运行 `python3 -I -B -S verify.py`（Python 3.10 或更新版本）。
不需要网络、权重或原始轨迹。应看到 Moving 三种上界排除数 67、36、9 / 72；
OpenDD 330/606 对 57/606；WEPDTOF 0/45 同时通过风险和效用检查；Songdo 联合策略 0/480 放行。
v4 另复算固定次数和位移后 OpenDD 的 210、60、42 三个关键计数和地点区间。
v5 保留这些结果，并复算成对偏好、等步数续训和风险内收益续训。
v6 新增编辑交换界限：零交换 60、至多一次 178–183、不限次数 210–212，另复算三个描述性收益门槛。
v7 保留这些界限，并将至多一次的范围精确收紧为 181；另复算同录像随机对照。
v8 增加地图和邻居消融：在同样 101 次交换上复算正确对齐 47 个双通过、
随机期望 51.21，以及对应的地点置信区间。主文仍为 v179。
四种冻结方案中 197/480 事后有合格方案的补充分析，从项目根目录运行：

```sh
python3 -I -B -S scripts/audit_m3w_songdo_frozen_menu_capacity.py
```

这不是重新推理或读标签；197/480 不是可部署成绩。

## 2. 本机：来源核对与真实检查点复跑

使用 Apple Silicon 原生 `.venv-pytorch/bin/python`。先核对解释器架构、Torch 和 NumPy；
不要用 Rosetta 下的 x86_64 Conda。训练入口设 4 个 Torch 计算线程、单进程采样，
不创建 DataLoader 子进程。若要换硬件或重建环境，先用冻结检查点跑通，再解释训练日志。

```sh
.venv-pytorch/bin/python -c 'import platform, torch, numpy; print(platform.machine(), torch.__version__, numpy.__version__)'
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_paper_evidence_lineage.py
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/run_m3w_recording_count.py --replay
```

来源核对应找到 7 个 OpenDD 地点、202 段录像和 3,395 个录像内不重叠查询；
OpenDD 检查点复跑应完成全部 21 个地点×种子组合，终态 `complete`，且 `updates: 0`。
它会读本机已经使用过的开发目标，但不会打开封存的 DroneCrowd 确认标签。
Git 里的轻量文件不含私有权重和原始数据。缺少冻结权重就不能声称复跑成功。

要复算 v175 新增的 OpenDD 固定实际编辑次数诊断，用
[`audit_m3w_opendd_realized_stream_capacity.py` 和 `report_m3w_opendd_realized_stream_capacity.py`](../outputs/publication_readiness_2026_09/opendd_realized_stream_capacity_v1/README.md)。
需要本机冻结权重和原先预算上限计算的逐录像记录；终态应有 606 个视图、780 次实际编辑，
其中固定次数事后解 242 个通过，线上 57 个通过。这不是匿名 v3 包中的检查。

随后运行 `scripts/audit_m3w_opendd_count_motion.py`，在每段录像固定编辑次数，
同时限制累计峰值位移。事后重排分数 60/606、已知标签事后解 210/606；
只重排分数不使用标签，但能看全段分数，因此不是线上部署。
[数值与复跑](../outputs/publication_readiness_2026_09/opendd_count_motion_v1/README.md)。

新训练的评分头用 `scripts/run_m3w_opendd_prefix_value.py --replay` 核对 42 个检查点，
断点恢复时去掉 `--replay`。固定预算结果为旧头 60、只学增益 48、同一联合头不带惩罚 57、
带惩罚 42。它没有通过开发对照，不能替换旧策略。
[训练与负结果](../outputs/publication_readiness_2026_09/opendd_prefix_value_v1/README.md)。

等预算方案排序用 `scripts/run_m3w_opendd_equal_spend_preference.py --replay` 复核。
同样的 780 次编辑，旧分数 60、独立分类 54、成对排序 59 个双通过；
排序增加了风险通过数，但没有增加同时满足收益和风险的方案。
[完整对照与训练记录](../outputs/publication_readiness_2026_09/opendd_equal_spend_preference_v1/README.md)。

风险内收益续训用 `scripts/run_m3w_opendd_risk_conditional_utility.py --replay`。
普通续训 55/606、风险内收益续训 58/606、旧分数 60/606；每组仍是相同 780 次编辑。
[训练与区间](../outputs/publication_readiness_2026_09/opendd_risk_conditional_utility_v1/README.md)。

交换范围用 `scripts/audit_m3w_opendd_exchange_frontier.py` 和
`scripts/report_m3w_opendd_exchange_failures.py`。第一条在旧方案周围找事后最优，
第二条分解原风险与收益失败。结果的上下端是已验证与未排除的计数，不是置信区间。
[完整计数与说明](../outputs/publication_readiness_2026_09/opendd_exchange_frontier_v1/README.md)。

排序器用 `scripts/run_m3w_opendd_one_exchange_ranker.py --replay` 复核；
`scripts/audit_m3w_opendd_one_exchange_ranking.py` 穷举 7,198 个交换，复算同录像随机对照。
不重新训练，不使用未打开的外部数据。[训练与读数](../outputs/publication_readiness_2026_09/opendd_one_exchange_ranker_v1/README.md)。

地图和邻居对照用 `scripts/run_m3w_opendd_exchange_context.py --replay`；
`scripts/audit_m3w_opendd_exchange_context_matched.py` 固定相同的交换视图并计算精确随机期望。
[输入、训练和复算](../outputs/publication_readiness_2026_09/opendd_exchange_context_v1/README.md)。

OpenDD 入口每 100 步原子保存模型、优化器、采样状态和训练身份。仅在确认既有
PID 退出、数据和注册文件一致后，才用原命令去掉 `--replay` 恢复训练；不要删除有效
检查点。恢复后先看 `heartbeat.json` 与实际日志，再核对结果及哈希。

## 3. CREATE：先查已有作业，再决定是否提交

主文这组复算在本机完成，不需重新排队。用既有获准的 CREATE 连接时，先检查
M3W 目录、配额和记录中的作业 ID。`squeue -u "$USER"` 查活动任务，
`sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed` 查终态。远程连接失败
不是作业失败，不据此取消、重启或重复提交。历史四个任务的核对结果见
[CREATE 回执](../outputs/publication_readiness_2026_09/create_accounting_recovery_20261108/README.md)，
它不证明当前有空闲 GPU。

仅对另一个确有需要、且划分和环境已冻结的 M3W 实验，在计算节点提交经核对的作业脚本。
保留 job ID、输入哈希、环境、输出目录、日志与检查点。恢复时用同一训练身份和采样状态，
先核对旧任务确实终止；`COMPLETED` 后仍需检查预测、训练步数、泄露边界和结果。
现有 [CREATE 时序实验手册](../outputs/publication_readiness_2026_09/european_temporal_auxiliary_v1/create_reproduction_zh.md)
给出已验证的项目路径、作业脚本和恢复示例，不是本轮的新提交命令。

## 4. 判断结果能支持什么

Moving 与 OpenDD 是开发证据；WEPDTOF 是冻结外部评价；Songdo 是同一采集日内的
分组校准与确认；DroneCrowd 只报告校准。三种子是同一录像的重复运行。
原 2% 是录像内风险条件，不是总体失败率；零放行不等于通过。保留原坐标与步数，
不把结果写成秒级、米级或物理安全。论文和复现包的构建通过不等于确认了部署收益，
也不等于已经提交。
