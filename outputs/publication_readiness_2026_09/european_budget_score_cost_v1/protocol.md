# Continuous Costs With Original-Budget Score Supervision

## Material Passport

Cached_verified: the complete matched-bank54head experiment and its separate
exact72group inference replay. Its new-bank policy is worse on known average
than the old bank and supports only37/444required recording-risk views.
Actual neural use has a small known-label increment,not a safe full policy.
Fresh_run in this experiment:162real cost fits and fixed development readout.
Not_run at registration:pilot,complete training,readout,independent outcomes.
No new trajectory producer or proposal bank is trained or selected.

The previous goal turn is progress:it completes real learning and rejects
candidate-only replacement as the sufficient repair. This experiment changes
the learning objective,not the split,risk tolerance,decoder or forecast bank.

## Hypothesis And Distinction From Previous Attempts

The deployed rule checks H <=0.02R and H_easy <=0.02R_easy. The current
regression estimates their five components with separate pointwise errors.
An optimistic denominator and an underestimated numerator can combine into
an optimistic decision even when aggregate moment loss falls.

Previous shared-reference heads tied the reference across actions. Previous
signed-score regression discarded identification of separate cost moments.
Previous teacher-region losses matched fixed selected means. This experiment
does none of those:retain the full five-moment and ranking objective,head,
input and causal decision interface;add only a coupled score-residual loss.
The symmetric-score arm isolates coupling from directional asymmetry. These
are fitting repairs and established statistical ideas,not a novelty claim
for asymmetric regression itself or a conformal guarantee.

## Fixed Three Objectives

Let q=(H-0.02R,H_easy-0.02R_easy),with supervised q* computed only in the
loss from known fitting costs. For each channel use the unchanged physical
fitting RMS scale s=sqrt(rms_H^2+(0.02rms_R)^2). This is numerical scaling,
not a standard error,covariance estimate or confidence bound.

1. original:the unchanged asymmetric five-moment MSE plus within-row ranking.
2. symmetric_score:original plus mean((q-q*)/s)^2 over legal known actions.
3. upper_score:the same added term,weighted4when q<q*,otherwise1.

The auxiliary weight is1,fixed before results. Under-harm weight4in the
original objective is unchanged. Every row is normalized by its legal known
action count,then averaged using the original sampler. Unknown future rows
remain in that past-only sampler with zero observed-cost loss. No oracle
selection,label-availability or future endpoint becomes an inference input.

## Full Matched Matrix

Use all6original contexts,3head seeds17/29/43,3banks and3objectives:
162real fits,2,000updates each,324,000primary updates. All54original arms
must exactly reproduce previous full scientific states. All banks and all
contexts remain,including physical_only,old_neural,new_neural;the actual
producer seed43,462causal fields,30,405parameters,width64,batch128,
learning rate0.001,AdamW,clip5 and ranking weight0.5stay unchanged.
Reuse original initialization,RNG,row draws,normalization,labels,physical
candidates,coverage eligibility and all three frozen past-only banks.

Nine-cell real first-context pilot:each objective/bank10+10/direct20exact
recovery,plus three unchanged original20step controls,420actual updates.
Require nonzero real updates,unknown draws with zero unknown observed loss,
and matched initialization/samplers. Use the measured speed/RSS before full
training. Do not replace the full run with the pilot. Independently refit
the first full new-bank upper_score head for another2,000updates.

Native arm64 .venv-pytorch,CPU4/interop1/workers0,no resource probing.
Checkpoint every100updates,heartbeat every50,explicit--resume,process lock,
12h boundary,10GiBfree-disk reserve,512MiBprivate checkpoint cap. Do not
restart a healthy task or reduce steps because it is slow. Commit all162
endpoints before policy scoring;retain failure and interrupted checkpoints.

## Fixed Readout And Interpretation

Primary contrast:upper_score new_neural minus original new_neural.
Keep symmetric_score as a mechanistic control and old_neural original as
the established policy comparator. Report every bank/objective,not the
winning seed or a post-hoc optimum. All216context/source/seed/objective
groups,all12localities and all74required recordings remain. Reproduce all
72original result groups and old inference hashes exactly.

Keep full and same-count utility,actual-path known/lower/upper paired
contrasts,hard/easy,action/neural intervention counts,common-union regret,
cost residuals,recording means/tails and finite/smoothness proxies. Retain
same-new-head physical restriction to isolate actual neural use. Empty
same-count comparisons remain not_estimable and do not count as safe zero.
Do not select a deployment model or threshold from these exposed fitting
outcomes. Retain3,000locality resamples,seed20261029,after context/seed
averaging. Paired missing-outcome contrasts use causal path disagreement
and the original known floor-reference mass,not imputed trajectories.

Progress gates:complete162fits;exact54original states and72original readouts;
complete upper_score new-policy recording risk under unchanged2%all/easy
rules;positive primary known and completion-lower CIs;positive completion-
lower contrast against original old_neural;positive completion-lower
actual neural-use contribution. Undefined support remains undefined.
No gate alone establishes independent confirmation or submission readiness.
Freeze initial full readout,then repeat every policy inference and metric
in a separate process. A null or negative result must not open independent
selection/calibration/confirmation or controller-held outcomes.

SOURCE is historically exposed supplied detector-silver bbox midpoint pixels,
obs8/pred12/stride12raw frames,not historical SDDt50,human gold,verified live
sensor-as-of,metric/seconds,true3D or foundation evidence. Scene-joint
decision training is not part of this single-factor objective experiment.
No Stage5C execution,SMC,independent-role or deployment change.
