# Donor-Easy-Constrained Risk Learning and Actual Policy Readout

## Material Passport

Prospective,registered before real fitting. One fixed objective change,no sweep.
The preceding coupled-support run repaired moment consistency but lost original
easy risk MSE5.69% and mean-alpha0.47%. It did not evaluate the revised actual
policy. This experiment tests whether donor objective conflict explains part of
the failure and whether its repair has useful policy consequences.

The user explicitly confirmed full-policy safety versus matched-utility separation
and asked to prioritize paper-relevant method evidence rather than accumulating
defensive proxy requirements. Accordingly,risk MSE is SECONDARY diagnostic here,
not a new admission gate or reason to withhold the fixed policy readout. Old
experiments and their failed preregistered screens remain unchanged. This is
not permission to relax2%,drop scenes or tune test outcomes.

## Estimator and Exact Contrast

Use the same72 exposed original-TRAIN inner-fit views,380 causal features,
32 selected other-recording donors,original normalizer and joint quadratic
prediction. Reconstruct each query's nested donor error e and direction d from
the previous experiment,checking original alpha and prediction hashes exactly.
Donor evaluation weights give each chosen recording equal total weight.

For each channel c (all/easy),L_c(a)=sum(w*(e_c+a*d_c)^2). The previous alpha
minimizes L_all+L_easy on[0,1]. A conflict is a donor fit with easy loss increased
>1e-12 while combined loss is non-increasing within numerical1e-10 tolerance.
This is an observed donor-training tradeoff,not proof of query failure causation.

Primary:keep the same combined objective,constrain L_easy(a)<=L_easy(0).
Let C=sum(w*e_easy*d_easy),V=sum(w*d_easy^2). The feasible interval is
[0,clip(-2C/V,0,1)] if V>0;V=0 gives[0,1]. Thus the constrained optimum is
min(previous_alpha,easy_cap). With no nested recording support usealpha0.
No added coefficient,threshold search,query label or future knownness is used.
Donor preservation does not guarantee query or deployment preservation.

Six fixed arms:origin,parent_blend,easy_only (minimize donor easy loss alone),
record_mean_easy,shuffled_easy(seed20261029),easy_constrained PRIMARY.
Mean/shuffle preserve per-query-record alpha mass,not action count. All outputs
are convex blends of original and joint cost;B/R/ER stay frozen. Query recording
is excluded from donor labels and normalization. Nested donor recording is
excluded from local supervised fit/source counts. Upstream models remain
exposed,so this is NOT whole-pipeline OOF.

## Actual Policy Comparison

Apply each risk estimator only to the original causal query subset;all other
rows retain original predictions. Re-evaluate the EXISTING fixed2% action rule
on the complete eligible fitting population,without outcome filtering:
benefit>harm,H<=.02R,EH<=.02ER. No new policy threshold is fitted.

Compare primary versus origin and parent_blend under full and per-record/frame
matched action counts using the existing deterministic ranking and completion
bounds. Report every full-policy view and recording,selected unknown rows,
all/easy positive-harm upper ratios,easy degradation,net utility lower mass and
paired completion lower utility normalized by the full known reference mass.
Full safety budgets are unchanged. Undefined full-policy risk is not a pass.
Empty matched controls retain zero utility contribution where defined;their risk
is explicitly not estimable,never safe zero and never an additional safety gate.
No view/comparator deletion. All six arms remain reported,not just the winner.

Per-record budget support is separated from the legacy joint flag that also
requires positive utility. Do not invent a requirement that every recording
must make a positive utility contribution. Report both quantities distinctly.

This fitting readout is a method falsification,not an independent statistical
or deployment claim. No confirmatory CI from repeated/exposed fitting windows.
No new neural forecaster. Independent roles stay closed. No automatic deployment
or independent-role opening regardless of fitting MSE or utility outcome.

## Diagnostics and Execution

Report all/easy donor losses separately,before/after,conflict count,alpha reduction,
query MSE and policy effects by source/context/head seed. Keep all62902 query
occurrences,including1567unknown. Head seeds are not independent forecaster
replications. No fresh model-selection loop after reading query outcomes.

Commit code,tests,protocol and registration before pilot. Pilot checks controls,
constraints,policy and writer paths and runtime;then complete72views and exact
replay. Separate arithmetic checks and focused regression tests suffice;do not
rerun unrelated unchanged historical suites. arm64 CPU4/interop1/workers0,
10GiB disk reserve,12h bound,checkpoint/heartbeat/lock/resume. Authorized original
TRAIN packets read-only;no new scheduler job or data upload. obs8/pred12,
rawstride12,pixel/local detector-silver. No metric/seconds/gold/physical-safety/
true3D/foundation or submission-readiness claim. Stage5C/SMC off.
