# Positive-Support Conditioning: What Changes and What Does Not

This is an optimizer-conditioning experiment, not a relaxed risk requirement,
new calibration theorem, or claim that the original failure has a unique cause.
The protocol and all training endpoints are frozen separately from the readout.

## Same Feasible Set

For required fitting recording r, let S_r(theta) be selected genuine known-easy
reference mass averaged over all its rows. Let m_r>0 be the smallest positive
eligible easy-row reference, divided by the recording row count. Exact floor
aliases are ineligible; they cannot supply artificial reference support.

The third constraint is S_r(theta)>=m_r. The original implementation uses
g_old=(m_r-S_r)/s_r, where s_r is the full recording easy-reference scale.
The controlled repair uses g_new=(m_r-S_r)/m_r. Since both denominators are
strictly positive, g_old<=0 if and only if g_new<=0. The two all/easy harm
constraints and their2%tolerance are unchanged.

For a fixed dual lambda, g_new=(s_r/m_r)g_old. The repair therefore multiplies
only the third effective primal-dual coefficient by s_r/m_r and updates that
dual using g_new. It changes optimization dynamics, not the feasible set.
Global gradient clipping can affect the resulting full parameter update;
identical feasibility does not imply identical optimization or convergence.

## Measured Weak-Pressure Mechanism

With nonnegative S_r and a zero-initialized support dual, the original update
lambda_{t+1}=clip(lambda_t+eta*g_old,0,L) implies

```
lambda_T <= min(L, T*eta*m_r/s_r).
```

This bound follows because every increment is at most eta*m_r/s_r; negative
increments and clipping cannot increase that upper bound. It is not a lower
bound on the multiplier needed to change a decision.

The old frozen fitting states verify the bound for every required record/head
view. Their median empty-support excess is0.00029222680313978344; in372/444
views per bank,500updates cannot raise the support dual to1. This is evidence
of weak relative numerical pressure, not proof that crossing1 guarantees a
feasible solution. With the new unit, an empty decision has g_new=1.

## Controls and Remaining Failure Modes

The disabled-conditioning implementation reproduces all scientific fields of
the original full500-update mixed-bank seed17checkpoint exactly, excluding
the declared registration/runtime identifiers. All three real pilots pass
interrupted5+5 versus direct10recovery. These are engineering controls, not
scientific efficacy results.

The primary trial keeps model capacity, causal descriptors, initialization,
action bank, record populations, optimizer, step count, temperature schedule,
all/easy risk budgets and evaluation roster fixed. It has54fits and27,000
full-gradient updates. The trajectory producer and moment predictors stay frozen.

Straight-through categorical gradients need not optimize the hard constrained
problem globally. Discrete choices, temperature saturation, gradient clipping,
competing constraints, causal feature insufficiency, domain shift and missing
outcomes remain possible failure mechanisms. Even a fitting-feasible solution
would not establish unseen-record safety or independent generalization.

Read out every fixed endpoint. Do not select a favorable seed, step or threshold
after viewing internal held records. Zero interventions leave selected risk
not_estimable; they do not pass the required recording gate. Oracle feasibility
uses labels and establishes no inference-time learnability. Independent model
selection, calibration and confirmation remain unopened. Stage5C andSMC stay off.
