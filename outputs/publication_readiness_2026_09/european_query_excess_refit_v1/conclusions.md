# Query-Aggregate Risk Supervision Did Not Repair Selection

Fresh training completed: 216 matched Torch risk heads, 432,000 updates.
Forecasters, protected floor, utility, causal features and source roles were
cached_verified. The registration preceded fitting; checkpoint freeze
`53c1b594` preceded decisions and action freeze `e4703640` preceded readout.
No independent selection, calibration or confirmation locality was opened.

## Main Finding

Pure query-aggregate signed-risk supervision is not supported as the repair.
At identical intervention counts in every current query, its ADE advantage
over pointwise supervision is **+0.009026% [-0.013970%, +0.041750%]**,
positive in only five of twelve locality contrasts. The interval crosses zero.
The selected-risk primary is structurally incomplete because ten fixed-parent
views have zero actions. This was disclosed before readout; it is not replaced
by the ADE result.

Under the same nominal predicted 2% budget, query-trained joint allocation is
**worse by 0.154680% [-0.305732%, -0.046464%]** than the matched pointwise
joint policy. This is not a count-matched comparison: intervention falls from
8.1507% to 7.4297%. Only two locality contrasts favor query training. It is also
worse than the frozen parent's joint policy, -0.120229%
[-0.227899%, -0.038637%].

Both arms improve mean ADE over the protected floor: 0.5201% for pointwise
joint and 0.3706% for query joint. Neither is certified safe. Pointwise joint
has 89/216 dependent views above the observed selected-harm budget and fourteen
undefined views; query joint has 98 violations and twelve undefined views.
The query model's worst easy net gain versus CV is +0.145638%, and no exact-CV
case is harmed. Net easy preservation is distinct from positive-harm control.

## Training Is Real, Generalization Is Not Repaired

All 108 pointwise heads and 104/108 query heads lower their own fitting-monitor
loss. This does not establish downstream lift. Query supervision has worse
held-source point-estimate MSE on all four reported loss metrics, including
its own aggregate metric: all-query MSE 0.078422 versus 0.077402; easy-query
MSE 0.001954 versus 0.001886. These are descriptive comparisons, not newly
registered significance tests. About 29.96% of held queries are single-agent,
where the two losses are identical.

## Decision

No deployment change, risk certificate, independent-confirmation claim, new
world-dynamics claim or submission-readiness upgrade. Retain both fitted arms
and all negative comparisons. Do not select the pointwise arm as a new deployed
winner merely because its mean ADE is higher.

The next repair should address selected-subset risk, preserve individual error
information and separate abstention/coverage from undefined selected-risk
ratios. It needs a new registration, not an outcome-driven amendment to this
experiment. The existing 2% screen and protected floor are not relaxed.

All intervals use 3,000 paired locality resamples after averaging dependent
views on twelve development localities. Image-local detector silver,
obs8/pred12 raw-frame stride12; no metric, seconds, human-gold, physical safety,
true3D or foundation claim. Stage5C and SMC remain disabled.
