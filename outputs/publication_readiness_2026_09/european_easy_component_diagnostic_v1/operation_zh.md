# 条件风险分量诊断：运行与复现

工作目录 `/Users/yangyue/Downloads/World`，本机使用原生 arm64 `.venv-pytorch/bin/python`。
这不是新模型训练：读取已有216个风险头，对108组拟合数据计算误差分解，不更新参数、
不搜索阈值、不产生新部署动作，不读取独立 selection/calibration/confirmation。

注册提交 `b68d127b` 在计算前已推送。首次作业 `37579489` 已提交，不能再次提交诊断。
数据在 CREATE 自有 M3W 目录，未复制4.91GB到本机，未修改 simulation 项目。
资源为 CPU4、16GiB、30分钟、单进程、workers0。首组耗时仅用于估算，不能冒充完成时长。

## 检查同一作业

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/manage_m3w_easy_component_diagnostic.py inspect
```

PENDING 是排队，不是失败；SSH查询超时不代表作业终止。先查同一 job ID，不重复提交。
每组完成即保存结果、心跳和哈希。若确实中断，保留完整组，核对原注册与输入后，
在有新调度凭据的恢复作业中给原 runner 加 `--resume`；不要修改旧回执。

## 正常结束后

首次作业 COMPLETED 0:0 且108组齐全后，才提交一次全组重放：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/manage_m3w_easy_component_diagnostic.py submit-replay
```

再次检查同一重放作业，确认 COMPLETED 0:0 和逐组精确匹配后再收集并独立核算：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/manage_m3w_easy_component_diagnostic.py collect
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/report_m3w_easy_component_diagnostic.py
```

上述不是无条件连续运行的命令。提交、收集和最终封存拒绝覆盖旧记录。
报告验证残差平方和交叉项、分区加总、逐场景查询加权和六种顺序的分量替换贡献；
后者另用子集公式核验，不把同一公式重复调用当成独立检查。

29项针对性测试覆盖核心分解和篡改拒绝，并非全部历史集成测试。拟合数据缺 signed
benefit，因此有用切换排序在本轮明确 not_run，不用正伤害标签冒充完整 gain 标签。
输出仅是拟合诊断；不能据此声称泛化、校准、部署安全或世界动力学提升。

观测8步、预测12步、raw stride12、图像局部坐标与 detector silver。没有米、秒、
人工gold、true3D、foundation或投稿完成声明。Stage5C和SMC继续关闭。
