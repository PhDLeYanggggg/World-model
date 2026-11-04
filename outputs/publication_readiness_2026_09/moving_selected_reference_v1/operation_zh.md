# 固定选择区域复核

```bash
.venv-pytorch/bin/python scripts/audit_m3w_selected_reference.py run
.venv-pytorch/bin/python scripts/audit_m3w_selected_reference.py verify
.venv-pytorch/bin/python scripts/report_m3w_selected_reference.py
```

run重新推理原神经头和新参考头，并核对原哈希；verify复用已核验私有参考向量，
仅重算标量。均不重新优化MILP或改变选择。空视图保留为不可估，不计为安全零。
表中非空视图均值只是事后条件描述，不是完整主要结果或部署收益。
私有缓存、权重和原数据不得提交。无新训练，Stage5C/SMC关闭。
完整逐录像聚合账本保留在本地groups/并单独忽略，不提交8.5MiB派生账本。
