# Policy-Region Cost Moment Training

## Material Passport

Prospective29October2026. The108-state frozen TRAIN audit and exact replay show
severe easy-harm underprediction inside current decisions,not universal H bias.
All230scene/mixed known easy-risk violations have an EH-dominant decomposition;
their median predicted/true EH mass is0.0131748573. Of92all-risk violations,
37are H-dominant and55reference-dominant. These are descriptive fitting views,
not independent confirmation or proof of one training cause.

## Avoided Duplicate Routes

The historical pointwise easy-harm deviance experiment already failed against
its quadratic control and selected-risk screen. Do not rerun it as a new idea.
Posthoc scalar/selected-set recalibration and scalar global moment matching
also have retained negative results. The compact19-input recording decision
head and its support-unit repair are different from training the current
398-input cost network. This trial is not a claimed new calibration theorem.

## Single Changed Training Objective

Keep the static-scene architecture26,309parameters,inputs,initial weights,
physical-fitting normalizers,all original sampled rows,optimizer,schedule,
2,000updates and inference gate unchanged. Add one fixed-weight1auxiliary
policy-region moment loss to the existing continuous/ranking objective.
The trajectory forecaster and all old cost-head endpoints remain frozen.

For eachcontext/seed/bank,use its old causal static-scene cost head as a fixed
teacher. Compute the original one-action choice on fitting inputs only.
Within eachfitting recording,define the known-label region selected by that
teacher. Precompute its mean normalized H,R,ER,EH targets from fitting labels.
Record identity is training-loss grouping metadata,never an input/embedding.
Unknown teacher-selected rows have no calibration label and remain unknown;
they are not safe negatives or excluded from later inference.

On the unchanged128-row minibatch,collect predictions at the fixed teacher's
chosen action for represented regions. Compare each region's predicted mean
to its full-fitting target mean using generalized I-divergence,averaged over
the four costs and represented regions. This is aggregate conditional-moment
training,not a replacement pointwise EH deviance or repeated posthoc scaling.
Only current fitting features are forwarded;no extra examples/optimizer draws.
The target centroid is a supervised fitting statistic,not independent calibration.

For normalized nonnegative t,p,use
D(t,p)=p+epsilon-t+xlogy(t,t)-t*log(p+epsilon),epsilon=1e-12.
The nonnegative-cost divergence is not a Poisson-count data assumption.
Its log-gradient can remain useful for small predicted mass. Synthetic gradient
checks are engineering evidence only. Minibatch region estimates introduce
variance and possible shrinkage;this is not exact all-record constraint training
or a guarantee under policy change. Missing teacher regions receive only the
original base loss. Do not invent positive support for them.

## Matched Execution And Readout

Sixcontexts x threeseeds x threebanks =54newfits/108,000updates. Require a
real all-bank10+10/direct20resume pilot and a full2,000update region_weight0
neutral that exactly reproduces the old scene head,Adam state,loss traces,
preprocessing and draws. Primary endpoints must match initial weights,normalizer,
sampler/Torch RNG,draw hashes and parameter counts. Training-region masks and
targets are hash-bound;no future/held normalization is allowed.

Freeze all54endpoints before first exposed inner-held readout. Compare current
and exact old scene policies,physical-only,mixed,neural-only,original binary
control and all fixed1/2/4harm sensitivities. Report known ADE,full completion
bounds,per-record all/easy2%risk,undefined support,unknown selections and
mixed-minus-physical/region-minus-parent paired3,000-locality intervals.
No post-readout best coefficient,threshold,seed or subgroup promotion.

Independent selection/calibration/confirmation and DroneCrowd stay closed.
Source inner-held is already exposed,not whole-pipeline OOF or a fresh test.
Native arm64CPU4/interop1/workers0,checkpoint100/heartbeat50,10GiBreserve,
12hhealthy cap with resume. No new trajectory producer,latent generation,SMC,
metric/seconds,true3D,foundation or deployment-success claim. All original gates
remain binding;empty selected risk is not_estimable,not safe zero.
