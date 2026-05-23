# Stage 25 Selector Failure Forensics

- 当前不是 true 3D，也不是 foundation world model；SDD 仍是 pixel-space raw-frame benchmark。
- 本分析重建 Stage24-style hard classifier，并结合 Stage24 报告解释 selector failure。
- baseline errors 只作为 labels/evaluation target，不作为 inference feature。

- Stage24 reported t+50 improvement: `-0.4326497359356306`
- Stage24 reported easy degradation: `11.328798263207801`
- reconstructed hard classifier t+50 improvement: `-0.32445384571760205`
- reconstructed hard classifier easy degradation: `10.603174214477132`
- selector regret: `29.23797504836798`
- harm over fallback: `22.394350187205376`

## Root Causes
- class_imbalance: `True`
- label_ambiguity: `True`
- train_val_test_distribution_shift: `True`
- horizon_mixing: `True`
- split_type_mixing: `True`
- agent_type_mixing: `True`
- confidence_calibration_failure: `True`
- fallback_policy_failure: `True`
- feature_insufficiency: `True`

## Conclusion
Stage24 failed because a hard best-baseline classifier optimized class labels, not regret. It over-switched low-margin/easy cases and had no conservative fallback gate.
