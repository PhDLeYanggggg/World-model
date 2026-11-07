# M3W 英文工作稿与实验复现

## 1. 本地环境

在项目根目录使用原生 arm64 的环境，不用 x86_64 Conda/Rosetta。先核对
架构，再检查 PyTorch 的实际运算。下面是运行时检查，不等于复现实验。

~~~sh
.venv-pytorch/bin/python -c 'import platform; assert platform.machine() == "arm64"'
.venv-pytorch/bin/python -c 'import torch; torch.set_num_threads(4); torch.set_num_interop_threads(1); x=torch.arange(64., requires_grad=True); (x*x).mean().backward(); assert x.grad.isfinite().all(); print(torch.__version__, torch.get_num_threads())'
~~~

正式校准使用四个 Torch 计算线程和零个 DataLoader worker。不要用多
worker 取代计算线程，也不要把这个小检查当作真实模型运行成功。

## 2. 重建工作稿与复核表格

~~~sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v12
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v12.py \
  --source-zip "$out/manuscript_source.zip"
cd "$out/build"
tectonic -X compile manuscript.tex
~~~

若 Tectonic 不在 PATH，使用桌面应用附带的 Tectonic 可执行文件，或
使用官方模板支持的 TeX 环境。若缺官方样式，将从 CVPR 2027 作者包
取得的 cvpr.sty 通过脚本的 --style 参数传入。脚本会拒绝与已核验
版本不同的样式。生成的 PDF 和源码包只保存在本地忽略目录。

正文排版和实验数值回放是两件事。按
[数值回放说明](../review_replay_v1/README.md)重算两张录像表和十组
联合对照，核对原始 2% 门槛和 3,000 次来源组自助抽样。两种操作
都不会重训模型、重跑外部确认或解除数据封存。

## 3. 查看、恢复 DroneCrowd 本地校准

~~~sh
base=data/stage_cvpr2027_experiments/dronecrowd_calibration_readout_v1
jq . "$base/job_state.json"
tail -n 20 "$base/calibration.log"
jq . "$base/heartbeat.json"
~~~

用 ps 核对状态文件里记载的两个 PID；日志一段时间不刷新不代表
训练或推理卡死。仅在确认原进程已退出后，从项目根目录执行：

~~~sh
.venv-pytorch/bin/python scripts/launch_m3w_dronecrowd_calibration.py start
~~~

监控程序按 infer、score、replay 的顺序执行。恢复时复用通过校验的
录像产物，不重复启动仍在运行的任务。只有 40 个校准录像的预测及
名册都冻结后，评分才会读校准标签。只有一个预先定义的策略同时
满足原 2% 规则、全部 120 个录像种子视图的参考支持及注册的正向配对
区间，才能运行 30 组独立确认。无参考分母不是安全通过。
详见[校准协议](../dronecrowd_calibration_readout_v1/protocol.md)。

## 4. CREATE 操作边界

本次工作稿和校准使用本地机器，没有提交 CREATE 作业。先阅读
[已核实的只读交接](../create_readonly_handoff_20260923.md)，不要把
旧的队列快照视作当前空闲资源，也不要使用 simulation 项目的目录、
环境、作业或凭据文件。

确需在 CREATE 运行时，先核实独立的 M3W 工作目录、合规数据路径、
允许的分区、环境和现有作业。只对已知的 M3W 作业 ID 使用
squeue -j JOB_ID 及 sacct -j JOB_ID 核对；连接异常或队列缺项
不能据此判断作业结束。经核实的作业脚本用 sbatch --parsable
提交一次，保存作业 ID、提交版本、输入校验和、日志及检查点路径。
不在登录节点运行训练，不用临时 GPU 分区规避限制。仅在确认原任务
已终止、检查点和配置匹配后恢复。完成后核对冻结产物、评分与回放
记录；调度器状态不能替代数值核查。
