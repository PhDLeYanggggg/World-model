# Moving-selection risk regression: exposed-development negative control

## Material Passport

This is a results component for research writing,not a submission-ready
manuscript or independent main result. All numbers refer to the fixed SOURCE
obs8/pred12/rawstride12 task,detector-silver local-image coordinates.

We test whether training on current rather than original selected actions
repairs selection-conditioned risk error. Thirty-six current-target risk
critics have matched architecture,initialization,source/query draws,2,000
updates,corrected loss units,point objective,loss weights,original context and
final decoder relative to36 fixed-query controls. The discrete current choices
are detached;no policy-gradient or differentiable optimization claim is made.
All forecast paths and original benefit/reference estimates stay fixed.

|Target/context|Known ADE gain over floor|Supported risk views /444|Known easy violations|
|---|---:|---:|---:|
|Fixed/local|8.563025%|2|396|
|Current/local|6.015268%|14|286|
|Fixed/selected-set|8.215347%|1|383|
|Current/selected-set|6.783336%|11|313|

The primary selected-set current-minus-fixed wholly-unknown-completion-lower
effect is-7.087372 percentage points,conditional95% locality interval
[-13.238339,-2.904036]. The known-label difference is-1.432011pp
[-3.335655,+0.608111]. Interventions fall from410,575 to137,815 repeated
views,but the complete-recording risk requirement remains failed. Improving
the number of supported views from1 to11 does not establish safe deployment.

Moving supervision also worsens the common fixed-original-set monitor and
all-action risk MSE. Current-query masked losses can change with the selected
population and have no supervised contribution from unknown/empty query sets;
their magnitude is not substituted for common-population prediction error.
This controlled negative result rejects the current implementation as a
sufficient repair,without proving impossibility for other policy-aware learners.

The positive current-target context-minus-local known contrast,0.768069pp,
does not survive as a supported same-count contribution:the reoptimized
common-count estimate is0.179028pp [-0.159545,+0.584688],while its completion-
lower difference is-1.589234pp [-2.936624,-0.477403]. Both variants fail the
unchanged original all/easy2% risk criterion.

All36 real fits,72 complete policy groups and two exact full scientific refits
are retained. A separate full replay exactly reproduces all72 groups,
144 fixed-query controls and the canonical summary,with its own verification.json;
this is not inferred from training agreement. The3,000 bootstrap
resamples use12 exposed localities after head/context averaging. Three critic
seeds are not three independent forecast pipelines. The repeated windows and
444 risk views are not independent observations,and adaptive search limits
all nominal intervals. There is no new trajectory-dynamics,independent
generalization,metric/time,human-gold,true3D,foundation or submission-ready claim.
Independent roles,Stage5C execution and SMC remain closed.
