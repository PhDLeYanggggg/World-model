# 本地核算与恢复

## 范围

本轮只重用已核验模型和已冻结策略，计算选中、剔除样本的风险构成。
不是重新训练神经网络，也不是独立测试。源录像逐一留出、源数据回代、
跨场景迁移三类结果分开保存。未来标签只用于已冻结动作的离线评价。

## 环境与运行

在项目根目录使用原生 arm64 环境，计算线程为4、inter-op为1，
DataLoader workers为0。入口在导入数值库前拒绝非原生Apple架构。

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_selected_pool_accounting.py register
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_selected_pool_accounting.py pilot
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_selected_pool_accounting.py source --resume
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_selected_pool_accounting.py replay_source
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_selected_pool_accounting.py transfer --resume
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_selected_pool_accounting.py replay_transfer
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_selected_pool_accounting.py report
```

注册文件应先提交再执行。原始输入、模型和校准参数留在私有数据目录，
不能仅凭公开汇总文件重现计算。每次运行核验上游代码、模型、参数与输入哈希。
修改方法后必须登记新版本，不得覆写已冻结的结果。

每完成一个源组或一批迁移组，保存包含PID与时间的heartbeat。
中断后使用source/transfer的--resume，已完成组保留并验证身份与哈希。
完整重放不跳过任何组，并要求数值结果精确一致。
process.lock用于排除重复运行，文件存在本身不能证明进程活着。

真实首组试跑14.10秒，峰值内存5,851,791,360字节，预计新增存储
38,560,960字节。该试跑时本地空间足够；后来机器剩余空间变化，迁移核算在
144/216组时触发10GiB保留空间限制。72组源域核算及重放已完成，不需要重跑。
已有CREATE任务37602475的完成记录属于上一实验，不是当前运行的作业。

## CREATE续算

资源修订不改变方法。216组固定输入直接从内存传到M3W专用目录，
没有本地大数组缓存。最初512MiB传输额度不足，另行登记为2GiB；
实际878,732,429字节。全部数据包哈希已验证。保留个人配额未知的限制，
遇到配额或写入错误就停止，不能删除其他项目文件。

当前作业37702155只提交了一次，配置4CPU、8GiB、一小时。首次观察为
PENDING，不是卡死或完成。不要重发提交命令，不要修改simulation任务。

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/manage_m3w_selected_pool_create.py inspect
```

只有调度器显示COMPLETED且退出码为0，完整重放及144组本地对照通过后，
才回收轻量结果。回收仍保留原来的本地磁盘空间要求。

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/collect_m3w_selected_pool_create.py
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_selected_pool_accounting.py report
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/report_m3w_selected_pool_findings.py
```

上述结果汇总命令在完整核算结束后才能运行；作业已提交不代表汇总已完成。
重建远程输入须使用已登记的export_m3w_selected_pool_create_v2.py，
旧export仍保留其原始512MiB保护限制，不应绕过。该操作不训练新模型。

## 本地空间不足时汇总

已补充登记一条只改变存储位置的路径。它先确认作业COMPLETED0:0，
核验216组、完整重放、至少144组本地精确对照和2592项原风险值，
然后在本机内存中执行原先冻结的汇总函数。数值计算不改写，
仍用相同的风险定义、缺失规则和3000次locality bootstrap。
原有针对性测试仍真实执行，完整数值汇总再重放一次要求字节一致。

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/report_m3w_selected_pool_memory.py report
```

只在作业完成后运行，不要在PENDING时反复尝试。代码/资源修订注册文件
已先行冻结；44项针对性检查通过，其中存储路径对照使用明确标注的
合成fixture，不能称为真实迁移完成。最多64MiB聚合产物写入CREATE的
M3W专用`memory_report_v1`目录，不生成本地大结果文件，也不提交新作业。
本地未来回收仍须满足10GiB保留空间，不能因此降低保护线。
远程只保存文件，统计计算不在登录节点执行。

## 独立数值复核包

源域图表现在有一个不依赖项目训练环境的复核工具：
`reproducibility/selected_pool_source/verify.py`，只需要Python和NumPy。
它从聚合量重算72个源域joint OOF视图的风险构成、504个图表数值和
3000次locality bootstrap。仍然是30个有效对比、42个无定义对比；
不把重复head当独立场景，不把无定义改为0。7项新测试通过。

以下命令在隔离子进程中执行复核，所有数值包只保留在内存，
不生成本地大缓存、不读取迁移结果，也不需要提交新的计算作业：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/package_m3w_source_reproduction.py
```

增加`--store-create`会把同一组4个文件写入已经授权的M3W目录：
`/users/k24101830/m3w/european_selected_pool_v1/source_reproduction_v1`。
已有文件必须字节一致，不允许覆盖其他实验。实际总量189,521字节，
远程文件哈希已经核验；此时本地磁盘仍低于10GiB保留线，未回收数值包。
单独使用完整包时，在包目录运行`python verify.py evidence.json`。

包内替换了locality/view标识，去除了作者、用户名和本地路径。
这不保证无法通过公开结果反查作者，不能叫作已通过会议匿名审查。
复核只覆盖聚合算术和bootstrap，不替代原始轨迹、teacher lineage、
特征因果性或checkpoint重放，更不等于完整论文可复现。

## 证据边界

29项针对性测试通过仅说明相应实现检查通过，不等于研究成功或全仓库测试通过。
逐组风险无定义、缺失标签和源支持不足都必须保留，不能记为风险达标。
只提供已经暴露的开发场景上的描述性/名义统计，不打开独立确认集。
部署策略不改变。Stage5C与SMC不执行。
