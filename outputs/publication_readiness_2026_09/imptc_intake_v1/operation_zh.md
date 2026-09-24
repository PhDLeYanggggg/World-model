# IMPTC 本地接入与复核

## 本轮完成了什么

已下载官方四段样例，逐条转换 142,361 条记录，并完成全量重放和独立计数实现的复核。
这是数据接入结果，不是模型训练、预测提升、独立确认或完整 IMPTC 实验。
59 项相关测试通过；没有重跑无关的旧版全套测试。

原始压缩包位于 `external_data/IMPTC_source_audit/`，派生数据位于
`data/stage_cvpr2027_experiments/imptc_intake_v1/`。两处均被 Git 忽略。
官方包内自带视频和图片，本轮没有解压、查看或上传这些媒体。

## 复现

在项目根目录使用原生 arm64 环境：

```bash
.venv-pytorch/bin/python scripts/fetch_m3w_imptc_sample.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_imptc_intake.py tests/test_m3w_ht21_intake.py tests/test_m3w_dronecrowd_windows.py -q
```

第一条会核对已存在文件的哈希，不重复下载。首次获取需加 `--download`。
下载保留 `.part`、错误日志和每30秒心跳；中断后使用相同命令续传，官方 MD5
和大小均吻合后才转为正式原始文件。下载耗时约两分钟，不是训练耗时。

本轮首次转换运行过：

```bash
.venv-pytorch/bin/python scripts/audit_m3w_imptc_sample.py
.venv-pytorch/bin/python scripts/audit_m3w_imptc_sample.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_imptc_support.py
```

转换及支持审计耗时3.555秒。脚本默认拒绝覆盖既有报告和缓存，因此已完成的
首次生成/复核命令不应直接重复执行：先核对 `verification.json`、
`separate_checks.json` 和源哈希。需要新版本审计时应保留旧结果、创建新版本目录，
不要删除旧报告伪装首次运行。当前脚本的输出目录是固定的，改版需同步登记。

## 使用边界

四段录像来自同一物理路口，不是四个独立场景。只有61条轨迹被源数据整体标为
行人，其余主要是车辆；这个标签仅用于审计，不进入推理特征。
没有创建训练/校准/确认划分，没有计算任何预测误差。

当前读取器不传入未来、中心速度、轨迹总长度、结束时间或全轨迹类别，缺失过去
不插值。未来缺失也不能改变样本是否属于审计总体。源数据本身经过跟踪滤波，
其上游时序因果性仍需核验，不能仅凭我们读取过去就宣称严格在线因果。

用户已授权日常审计由项目执行方负责，无须反复索要人工批准。遇到证据不足时
保留 `not_run` 与具体原因，并继续可以独立推进的工作；不把授权当作科学验证。
DroneCrowd仍封存；当前部署不变；不执行Stage5C，不启用SMC。
