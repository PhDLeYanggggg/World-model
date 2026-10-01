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
38,560,960字节，保留10GiB磁盘底线。该工作适合本地，没有新CREATE任务。
已有CREATE任务37602475的完成记录属于上一实验，不是当前运行的作业。

## 结果解释

29项针对性测试通过仅说明相应实现检查通过，不等于研究成功或全仓库测试通过。
逐组风险无定义、缺失标签和源支持不足都必须保留，不能记为风险达标。
只提供已经暴露的开发场景上的描述性/名义统计，不打开独立确认集。
部署策略不改变。Stage5C与SMC不执行。
