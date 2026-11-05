# Paired-Loss Gradient And Decision Audit

This posthoc own-B diagnostic follows the completed negative paired-decision
experiment. It does not change any trained model, inference threshold, data
role or evaluation rule. Original2% all/easy risk, signed easy-net and support
requirements remain; no identity certification. No new V/H or reserved data.

Analyze all nine frozen joint/point/rank models using their entire own-B
fitting population and exact saved query-frequency weights. Decompose weighted
gradients into risk regression, decision supervision, joint supervision and
the extra paired point/rank term. Use fixed cyclic pair offsets0 and1999,
report both without selecting an offset. These are final-model gradients,
not gradients along the full training path or a proof of its causal mechanism.

For the three rank models, compare four disposable saved-AdamW updates:
base objective; base plus rank; rank with only its risk-parameter gradient
removed; rank with its conflicting risk-gradient component projected away.
The base risk/decision/joint gradients, optimizer moments, clipping, rate and
weight decay remain fixed. The non-risk auxiliary gradient is unchanged.
The projection is one-sided, specific to the extra rank term; it is not a
complete reproduction of PCGrad or an empirical2% safety guarantee.

Report full-fitting MSE, signed regret and joint decision cost after all12
disposable updates. No updated checkpoint is persisted. A protected-gradient
training follow-up is supported only if at least2/3 rank contexts show negative
risk/extra-rank gradient dot product at both fixed offsets, and at least2/3
protected steps lower full-fitting risk MSE than the full-rank step without
increasing joint decision cost. Do not use a held outcome to change that rule.
This gate permits a controlled training hypothesis, not scientific deployment.

Earlier all-versus-easy regression audits did not support broad task conflict.
Earlier norm balancing on European SOURCE data failed matched utility. Those
results stay negative; this study tests the newly introduced paired objective,
not the already-rejected generic explanation. Local cached assets suffice;
use arm64 CPU4/inter-op1/workers0 with process lock and heartbeat. Results and
exact replay are separate from independent confirmation. Native obs8/pred12
image coordinates, not metric, seconds, true3D, foundation or human gold.
Stage5C/SMC remain disabled.
