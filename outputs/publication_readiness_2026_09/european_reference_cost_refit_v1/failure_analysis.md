# Cost Repair: Failure Taxonomy And Interpretation

This follows the complete registered readout. It does not select a new primary,seed,threshold or recording subset.

|Question|Measured evidence|Conclusion boundary|
|---|---|---|
|Sampling coverage repaired?|All36 fits have minimum4query visits and zero undrawn queries|Repairs sampling coverage;wholly unknown labels still provide no supervised gradient. Not a generalization claim.|
|Reference learning helps the registered decision target?|Known:+0.707816 [+0.238023,+1.192602];completion-lower:-2.031363 [-4.275018,-0.420831]|Use the lower-CI gate,not the favorable point estimate alone.|
|Reference prediction improves?|Normalized reference MSE change:-0.154861 [-0.230003,-0.097589]|Fitting error is not independent calibration;ratio or sign errors may remain.|
|Reference-target consistency?|Locked:894975/894975;learned:894975/894975 multi-action row views have differing predicted R|The actual reference target is action-invariant;row-shared scaling has not repaired that structural inconsistency.|
|Implied candidate error coherence?|Negative expected ADE from moments:locked93505/5369850;learned169709/5369850|These are cost-model contradictions,not negative measured trajectory errors. They increase despite lower reference MSE;selected-policy causation is not established.|
|Benefit-reference ordering?|B>R:locked136156;learned251929 legal action views|The inherited coupled head does not enforce B<=R. Both positive and harm expectations still need representation;netting them away would hide risk.|
|Original full-policy risk?|Supported 9/444;undefined easy:9;known easy-risk violations:369|No full-policy safety pass from pooled gains or count-matched controls.|
|Easy accuracy tails?|Worst known degradation:+17.170690%;above2%:22|Do not hide individual recording harm.|
|Capacity attribution|262additional active parameters in the learned arm|Not fully active-capacity matched;the changed branch is not proof of a general architecture contribution.|
|Baseline continuation attribution|Both arms change coverage and continue the old fit with reset optimizers|Cannot isolate coverage alone versus extra updates relative to the historical parent.|
|Exact-zero reference explanation?|Locked R/ER zeros:0/0;learned:0/0|Multiplicative zero recovery is a theoretical limitation,but not an observed cause in these legal action views.|
|Missing labels|Original conservative bounds and full roster retained|Unknown disagreement is uncertainty,not observed harm or a fabricated zero.|
|Generalization|Exposed SOURCE fitting,one forecast producer|No independent confirmation,cross-dataset success or submission readiness.|

Fitting advancement:False. Deployment stays unchanged.
