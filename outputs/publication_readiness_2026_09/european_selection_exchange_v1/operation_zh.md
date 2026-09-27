# 冻结决策的收益与伤害分解

本轮不是新训练，也不是调整阈值。分析的是已冻结策略在相同当前画面、相同
切换数量下，换入和换出哪些预测，并核对收益差减伤害差是否还原既有 ADE 差。
未来标签只用于结果核算；未知标签保留在切换人数中，不当作零误差。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_selection_exchange.py --phase run --resume
.venv-pytorch/bin/python scripts/run_m3w_european_selection_exchange.py --phase replay
.venv-pytorch/bin/python scripts/report_m3w_selection_exchange.py
.venv-pytorch/bin/python scripts/verify_m3w_selection_exchange.py
```

本机原生 arm64，CPU4、interop1、workers0。每完成一组保存小型汇总和心跳，
中断后续跑，不复制大缓存。完整 108 组 fresh_run：PID36802，44.22 秒，最高
RSS8,445,181,952 字节；完整重放：PID36917，44.40 秒，最高 RSS8,208,580,608
字节。两次结果一致。保持至少10GiB磁盘余量。

CREATE 仅按已有批准配置执行只读队列检查，返回码0，无提交、无认证或环境修改。
本轮私有回执哈希：
`e57bdf86aa757992ad81bc98179eb751a686ac56840404c18aba1ab26bd2bec8`。

验证记录区分单元测试、完整重放、逐画面人数核验和来源级统计；不把这些技术
检查当作科学成功。旧全量测试与从 raw data 冷启动重建仍是 not_run。

查看 results.md、conclusions.md、verification.json。下一步测试收益感知的当前
画面联合选择，仍保留2%预算、真实伤害检查、独立确认数据封闭与禁止生成式执行。
