# Selected Costs: Easy Harm, Not Uniform Harm Bias

Descriptive TRAIN diagnosis of108frozen states/2,664record views,zero updates.
Every original action hash and per-record score is replayed. This export
summarizes verified frozen inference;it does not train or calibrate a policy.

## Mixed-Bank Pools

Medians use finite record-level ratios,not pooled windows. Repeated record/
context/head views are not independent recordings. All-legal and gate pools
repeat candidate rows and are not deployment policies.

| Family | Pool | Action views (unknown) | Median known all risk | Median known easy risk | Median predicted/true H | Median predicted/true EH |
|---|---|---:|---:|---:|---:|---:|
| no_scene | all_legal | 5369850 (113040) | 0.383620 | 0.589251 | 1.447308 | 0.939360 |
| no_scene | gate_passing | 100229 (1377) | 0.009861 | 0.019514 | 2.168178 | 0.028022 |
| no_scene | selected | 54296 (726) | 0.009916 | 0.021351 | 2.206267 | 0.026394 |
| scene | all_legal | 5369850 (113040) | 0.383620 | 0.589251 | 1.498049 | 1.026498 |
| scene | gate_passing | 134652 (1649) | 0.008885 | 0.021517 | 2.162030 | 0.020350 |
| scene | selected | 70561 (853) | 0.008614 | 0.024664 | 2.241466 | 0.021367 |

## Accounting On Known Violations

Observed minus predicted risk excess = harm underprediction + reference
inflation,with analogous EH/ER terms. Negative components alleviate error.

| Family | Event | Violating views | Harm component dominates | Median predicted/true harm | Median predicted/true reference |
|---|---|---:|---:|---:|---:|
| no_scene | all | 111 | 42 | 0.749408 | 2.937061 |
| no_scene | easy | 214 | 214 | 0.013821 | 0.547894 |
| scene | all | 92 | 37 | 0.537079 | 2.576502 |
| scene | easy | 230 | 230 | 0.013175 | 0.464233 |

## Interpretation

Global harm underprediction is not the universal failure:the selected
mixed-bank median H prediction is above its observed mass. Easy harm has
a different pattern. Do not repair every cost by a generic multiplier.
The exact decomposition measures where prediction error enters the gate;
it does not prove selection caused the error or identify its training cause.
Composition,rare easy events,small support and model misspecification remain.

The next single-factor training test targets the sparse easy-harm loss
gradient,retaining architecture,inputs,draws,optimizer and2%evaluation rule.
It is a hypothesis,not a passed gate. The known-label ratios here do not
replace full-completion bounds. Empty risk stays not_estimable. Independent
roles and deployment remain unchanged;no Stage5C orSMC.
