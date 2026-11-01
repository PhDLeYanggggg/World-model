# Measured Diagnosis And Next Controlled Experiment

## What Changed In This Complete Control

Known ADE gain over the same floor:+10.329562% -> +10.097661%.
Paired known effect:-0.231901 [-0.303124,-0.162510]pp;registered completion-lower:-0.455097 [-0.618925,-0.304167]pp.
A negative known effect means the failure cannot be attributed only to missing-outcome conservatism.
The projection bundles reference averaging with benefit clipping;this experiment does not isolate their individual effects.

|Normalized SOURCE MSE|Raw learned|Projected learned|
|---|---:|---:|
|benefit|+0.267517|+0.274885|
|harm|+0.095654|+0.095654|
|reference|+0.191049|+0.185460|
|easy reference|+0.335499|+0.323636|
|easy harm|+0.112226|+0.112226|

Reference sharing removes candidate-dependent reference values,and the benefit cap removes negative implied errors.
This is algebraic coherence,not accurate conditional costs. Reference MSE and policy utility need not move in the same direction.
H/H_easy are unchanged at every candidate;selected-action errors may change only because the chosen actions differ.

|Learned policy|Selected known views|Normalized selected H bias|Normalized selected H_easy bias|Known all/easy risk violations|Risk-supported recording views|Worst known easy degradation (%)|
|---|---:|---:|---:|---:|---:|---:|
|raw_learned_joint|393752|-0.012030|-0.008340|317/369|9/444|+17.170690|
|learned_joint|391684|-0.011828|-0.007910|309/360|9/444|+14.745066|

These selected-action biases are count-weighted descriptions of repeated SOURCE views,not independent calibration or a causal proof.
Negative harm bias is consistent with residual optimism on chosen actions. Actual positive-harm violations establish that this is not only a missing-label issue.
Worst-case missing-label bounds are not observed damage. Native cost sums are not pooled across incompatible source scales.

## A Useful But Insufficient Component Result

Projected learned joint minus its own independent policy:known +1.627091 [+0.892532,+2.407216]pp;
completion-lower +1.014536 [+0.426580,+1.678899]pp.
The native allocator fixes each query to its own independent intervention count. This is not a replacement primary or a safety result.
Post-hoc trimmed count-matched aggregate is not_estimable when a required view is empty;learned projection empty groups:3.
It cannot be reported as a successful matched-policy or zero-risk result.

## Next Action And What Is Not Yet Run

The current query-level training error in m3w_query_unit_critic.query_error uses frozen causal initial choices.
All legal known candidates still receive pointwise supervision,and cyclic sampling covers every query;
neither property makes the aggregate loss train on the critic's own newly optimized choices.
A focused next comparison should use one shared-reference coherent decoder in both arms,with equal capacity,draws and budgets.
Change only fixed-choice cost regression versus decision-aware constraint learning on current choices,while retaining all known-action supervision.
Use only available training outcomes for supervised violations;never substitute unknown targets with true zero harm or call a partial-label witness a true optimum.
Preserve the full cohort,2%contract and independent role locks. Include a disclosed published multi-expert/constraint-aware comparator before claiming novelty.
That training comparison is not_run here. This completed experiment is a zero-update intervention on frozen costs,not another neural training success.
A failed mean-and-cap postprocessor does not reject every coherently trained model. No deployment,Stage5C or SMC promotion.
