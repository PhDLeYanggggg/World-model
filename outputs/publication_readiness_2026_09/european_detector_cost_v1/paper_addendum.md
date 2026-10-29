# Manuscript Evidence Addendum: Detector History Is Not Sufficient

## Controlled Question

We test whether causal detection shape and quality histories improve the
decision to substitute a frozen neural trajectory for a strong motion floor.
The original398cost inputs are augmented by a64-dimensional zero-initialized
branch. Disabled,current-only,identity-noise and full eight-observation
histories share parameter count,query stream,loss,optimizer and fixed budget.
This intervention changes controller information,not trajectory dynamics.

## Source-Development Result

Across six original fitting contexts,three controller seeds and two candidate
banks,we fit144Torch heads with288,000primary optimizer updates.36disabled
endpoints reproduce the previous model/optimizer/RNG/trace exactly. A separate
complete history-head refit verifies scientific-state reproducibility. The
readout includes all288source/representation/seed groups. Uncertainty uses
3,000locality resamples after context/seed averaging over12localities,not
overlapping windows as independent samples.

Adding history increases known-label ADE gain over the floor from0.581%to
1.969%. Its paired known gain over disabled is1.388pp,conditional95%CI
[0.413,2.888],and it exceeds both current-only and identity-noise controls on
known utility. The most conservative missing-outcome contrast is not stable:
history-minus-disabled full lower CI is[-0.109,2.536]pp. Complete all/easy
recording-risk support remains42/444dependent views,with307known easy-risk
violations. The mixed-minus-physical known change is0.023pp,CI
[-0.033,0.091]; its completion-lower CI is[-0.521,-0.131]pp.

## Permitted Interpretation

The past detector feature block improves some fitting cost decisions but is
not sufficient for safe neural substitution. Observed utility cannot replace
post-selection risk or incomplete-outcome bounds. A confidence-only causal
mechanism is not isolated because the block also contains body-relative motion
and IoU. Positive fitting contrasts do not establish external transfer,neural
dynamics lift or deployment safety. The physical bank captures nearly all
observed mean improvement. Negative lower-bound contrasts do not identify a
negative true effect on missing outcomes.

These findings belong in source-development/negative-control material,not a
new independent main benchmark claim. SOURCE remains supplied detector-silver
midpoint pixels,obs8/pred12/stride12 raw frames,with unverified acquisition-as-of,
physical scale and effective timing. Independent roles and DroneCrowd are not
opened. There is no new JEPA,Transformer,generative rollout or SMC training.

## Remaining Main-Claim Gap

A useful forecast proposal must add learnable value beyond the physical bank,
and a learned policy must retain the complete approved safety contract. The
same causal detection histories have not yet been tested as inputs to a newly
trained trajectory producer. That is a distinct prospective experiment,not a
positive result inferred from this controller ablation. The current evidence
does not establish a CCF-A/CVPR submission-ready core method.
