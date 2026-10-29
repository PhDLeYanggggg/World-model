# Record-Constraint Diagnosis

Frozen TRAIN outcomes and checkpoint metadata only. No new model or evaluation cohort.
Counts repeat 74 source-qualified recordings across contexts and three policy-head seeds.

|Policy|Supported / required|Known all over|Known easy over|All upper over|Easy upper over|Undefined easy|Net easy over|
|---|---:|---:|---:|---:|---:|---:|---:|
|origin|16/444|331|164|408|330|73|329|
|cost_sensitive|84/444|346|54|346|70|14|0|
|record_balanced|131/444|289|39|290|59|9|0|

## Constraint Weight Direction

Only constraint penalties change. Utility weights and the total constraint weight remain fixed.
These slices describe final outcomes; they do not isolate a causal effect within each recording.

|Direction|Cost-sensitive supported|Record-balanced supported|Cost-sensitive known all over|Record-balanced known all over|
|---|---:|---:|---:|---:|
|decreased|54/72|55/72|7|9|
|increased|30/372|76/372|339|280|

## Dual Snapshot

Excess is just before the final optimizer update, duals just after it. Final safety uses the full readout, not this trace.

|Policy/channel|Positive excess|Dual at cap|Both|Maximum dual|Median positive scaled excess|
|---|---:|---:|---:|---:|---:|
|cost_sensitive/all|344|0|0|25.8845|0.004253155510591584|
|cost_sensitive/easy|77|5|1|100|0.001736974168206646|
|record_balanced/all|286|0|0|17.9144|0.0018795141421402248|
|record_balanced/easy|61|3|0|100|0.0010655158714143365|

All original recordings and the 2% budget remain. Empty or unsupported ratios are not safety passes.
This diagnosis supports the next training-only repair; it does not select a deployment threshold or prove generalization.
