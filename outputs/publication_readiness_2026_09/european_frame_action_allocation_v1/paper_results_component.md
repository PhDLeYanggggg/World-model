# Results component: matched current-query allocation

This is an exploratory methods/results component,not an independently validated
main paper or a submission-ready claim.

## Controlled design

We hold the seven candidate forecasts and18 trained cost heads fixed and vary
only current-query allocation. Queries group allowed target rows by recording
and current raw frame;they need not contain every visible agent. The primary
joint method matches the independent intervention count in every query and
maximizes predicted benefit minus harm subject to the same all/easy2% signed
budget criteria. Uniform-action and free-count allocations are retained as
secondary controls. No new predictor training or threshold selection occurs.

All72 groups complete and reproduce in a separate inference/solver/metric
process. The original parent predictions,actions and complete metrics agree.
The main run takes729.54s at10.92GB peak RSS;replay takes741.43s. Eight
postcheck-rejected primary solutions preserve the independent assignment and
are not described as exact-optimal solutions.

## Utility and protection

Known ADE gain relative to the frozen floor is1.904562% for independent,
0.626333% for uniform,2.552950% for matched joint and2.704414% for free joint.
The primary matched-joint increment is+0.648388pp,95%CI[+0.181208,+1.272930].
The wholly-unknown completion-lower increment is+0.506055pp,
CI[+0.071211,+1.107688]. Both primary contrasts are positive for all three
cost-head seed means. The total265,671 interventions are matched query by query.

However,recording-risk compliance falls from46/444 to43/444 required reused
views. Known easy-risk violations rise308 to324 and known all-risk violations
149 to231. Undefined easy support remains42. Thus predicted budget pooling
improves fitting utility but does not protect the complete strategy. The result
supports a narrower allocation hypothesis,not a safe deployed method.

The label-aware whole-record witness is freshly reproduced and satisfies
algebraic budgets in all37,532 seed-deduplicated context/query views. Only20,858
have defined selected easy support. Such opportunity cannot establish causal
learnability,complete-frame support or a matched-rate optimum.

## Uncertainty and limitations

Intervals use3,000 locality bootstrap draws after context/head-seed averaging.
They are conditional exploratory fitting intervals over12 already exposed
localities,not independent cross-scene confirmation.74 recordings generate
444 reused recording/context/seed views;the forecast producer seed43 stays fixed.
ADE uses available requested future positions. Full-roster FDE is not estimable
where a required locality lacks endpoints. Wholly-unknown completion sensitivity
does not bound missing interior positions in partially observed ADE labels.

No pairwise interaction model is trained. Predicted geometry summaries are not
physical safety measurements. A leave-neural-use-out ablation and an optimized
physical-only joint comparator answer different questions and must not be
conflated. The latter is still needed for a stronger neural contribution claim.
Supplied detector-silver top-down pixel/raw-frame data are not human gold,
metric,seconds-level,true3D or foundation-model evidence. No deployment,Stage5C
execution or SMC follows from these fitting results.
