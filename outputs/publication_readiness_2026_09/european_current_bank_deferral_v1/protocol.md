# Current-Bank Published Deferral Control

## Material Passport

Prospective SOURCE control after the complete own-decision cost study. Its
observed routing improves,but its registered completion-lower and recording
risks fail. The published logistic-deferral code already exists on an earlier
forecast bank;those numbers are not a matched comparator for the updated bank.
This experiment reuses its loss and causal choice code,not its old scores.

All149,510targets,74recordings,12exposed localities,72groups and source roles
stay fixed. Same obs8/pred12/rawstride12,seven current forecasts,462fields,
normalizers,causal initial query context and legal action availability. No
independent outcomes are opened. Three cost seeds share forecaster43.

## Loss Control

Train two query-conditioned score networks. Same30726active parameters,warm
encoder from each preceding learned-reference head,zero-initialized action/floor
outputs,fresh AdamW,full cyclic query draws and update budget. Query encoding
uses exactly the preceding causal features,cost anchors,disagreement envelope
and frozen initial selected-set context. The floor logit uses the same analytic
hidden pool as that reference branch. No future target or validity enters it.
Capacity is matched between these arms,not exact to the31119parameter critic.
The architecture/initial output parameterization differs from that critic.

ADE arm:softplus expected-ADE regression. TDEF arm:existing Mao/Mohri/Zhong
ICML2024Eq3logistic surrogate with coefficient sum(costs)-cost_j,not one-hot
oracle labels. This is an application adapter,not reproduction of their tabular
experiments or verification of a bounded-loss consistency theorem for native
partial ADE. Both use the same eight expert costs:floor plus seven candidates,
divided by each row's frozen causal scale. No label clipping or new normalizer.
Unavailable actions are tied to the exact floor score and floor cost in both
arms,not credited as interventions. Each complete query contributes its mean
known-row loss;the16-query batch averages these values. Unknown target rows
remain NaN,enter causal context and sampling,but receive no supervised loss.
An entirely unknown query contributes no target,not a fabricated zero cost.

## Policies And Registered Primary

For each arm,report raw independent score selection and a guarded version.
The latter filters candidates with the frozen own-choice critic's original
pointwise positive-gain/two2%risk rules. No threshold fitting or new calibration.
This is predicted feasibility,not guaranteed actual recording risk. Published
scores are not native ADE utility,and are never passed off as native utility
for an integer allocator. Keep frozen own independent/joint and floor controls.

Primary:TDEF_guarded minus ADE_guarded FULL completion-lower utility,lower95%CI>0.
Also require TDEF_guarded minus frozen own joint lowerCI>0,own-floor lowerCI>0,
and all original recording all/easy risk and easy-preservation requirements.
Raw contrasts and within-arm guarded-minus-raw are secondary,not alternate
primaries. Post-hoc common-count comparisons use within-policy causal score
margins for trimming and measure utility only. Any required empty view stays
not_estimable,not a safe zero. The original2%contract is unchanged.

Report known ADE,full-roster FDE availability,completion bounds,all recording
risks,easy tails,hard/easy subsets,action use,regret,unknown selections and pair
contraction proxy. No metric/seconds,physical-safety or world-dynamics claim.
Bootstrap3000paired localities,seed20261101,averaging repeated context/head
views first. These are conditional SOURCE fitting intervals,not confirmation.

## Execution And Reproducibility

Native arm64 .venv-pytorch,CPU4/interop1/workers0,no Torch resource probing.
Pilot:400actual updates with interrupted/direct100-step recovery in both arms,
complete causal prediction/choice recovery,no new policy outcomes scored.
Use the same minimum2000updates/four largest-source cycles:208320primary updates
over36heads. Repeat both7168-update first-context/seed17fits from scratch,
adding14336updates. Compare complete scientific state,excluding elapsed time.
Every query must receive at least four visits;paired draws must match the
preceding own-choice experiment. Checkpoint/trace50updates,explicit resume,
phase lock,12hphase boundary,10GiBdisk reserve,1GiBprivate checkpoint cap.
Measure the real pilot before assigning local versus CREATE compute.

Evaluate all72groups and separately repeat new-score inference and saved-action
scoring. Frozen parent full and recording scores must remain exact. Public
scalar reports must reconstruct without private input reads. No raw data,
features,caches,weights,images/videos or third-party files enterGit. No
deployment,Stage5C or SMC is authorized by a SOURCE result alone.

Primary source: [Regression with Multi-Expert Deferral](https://proceedings.mlr.press/v235/mao24d.html).
