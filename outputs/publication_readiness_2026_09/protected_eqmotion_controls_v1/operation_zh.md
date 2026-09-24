# Full EqMotion 匹配对照：复现与恢复

本轮已完成12个森林训练、12个既有神经头核验、完整决策复算和独立算术
检查。当前没有运行中的实验进程，不需要为了补报告重新训练。

## 资产边界

已有18个双场景排除预测器、12个外层预测器和12个所需神经成本头。
本轮核验复用，不重新训练预测器。仅补12个 full-forecast 森林头；
既有 ramp/uniform 森林不能冒充这个对照。所有源场景已用于开发。

## 执行

在 `/Users/yangyue/Downloads/World` 使用原生 arm64 环境：

```bash
.venv-pytorch/bin/python scripts/run_m3w_protected_eqmotion_controls.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_protected_eqmotion_controls.py --resume
.venv-pytorch/bin/python scripts/run_m3w_protected_eqmotion_controls.py --evaluate
.venv-pytorch/bin/python scripts/run_m3w_protected_eqmotion_controls.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_protected_eqmotion_controls.py
.venv-pytorch/bin/python scripts/report_m3w_protected_eqmotion_controls.py
```

CPU4、interop1、workers0。每16棵树保存原子检查点；heartbeat、events、
本地日志位于 `data/stage_cvpr2027_experiments/protected_eqmotion_controls_v1/`。
运行锁禁止同一实验重叠启动；中断后保留目录，用相同 `--resume` 继续。
不要改变冻结脚本或配置后继续旧检查点。慢不等于卡死。

所有12个森林完成后才能统一评估。所有决策先冻结，后读取外层标签。
旧神经分数与选择必须逐行一致；旧简单动作与 Transformer 汇总必须复现。
逐行结果、目标标签、缓存和权重不进入Git。

最后一条命令只读取汇总及核验记录，独立检查87个配对差值和置信区间，
生成完整对照CSV、轻量汇总及训练loss表。算术核验通过不代表科研假设通过。
本轮73项针对性测试通过，未重跑全部历史测试：

```bash
.venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_eqmotion_contrast_reporting.py \
  tests/test_m3w_protected_eqmotion_controls.py \
  tests/test_m3w_protected_motion_controls.py \
  tests/test_m3w_external_cost_bank.py \
  tests/test_m3w_eqmotion_cost_data.py \
  tests/test_m3w_bounded_cost_head.py \
  tests/test_m3w_protected_motion_verification.py \
  tests/test_m3w_native_metrics.py
```

## 解读

主要指标仍为四物理场景相对CV的ADE改善率等权平均，观察8步、预测12个
原始标注步，不是 t+50，也不是秒或米。新计算不意味着独立测试；三种子
不能替代独立场景。Bootstrap按四物理场景抽样，不按重叠窗口抽样。

相同切换数量对照是离线排序诊断。必须同时查看easy、hard、缺标签增益
界限和尾部，不能只按平均提升宣布模型可部署。旧 EqMotion refit 主检验
失败仍保留，本轮不会把它替换成一个有利的次要结论。

DUT不重调参，不重新读取预测结果；DroneCrowd确认数据继续封闭。
CREATE本次认证失败，未修改任何凭据或其他项目。本轮小型控制头在本地
即可完成。没有新部署，没有 Stage5C，没有SMC。
