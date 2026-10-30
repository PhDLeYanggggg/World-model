# Methods/results addendum: admission-gradient repair

This is a reproducible negative-result component for the working manuscript,
not a completed submission or an independently confirmed method contribution.

## Question and relation to existing work

When a gain/harm head underestimates risk on the actions it selects, is a
current-admission training penalty ineffective because it is seldom active,
or because its gradient contribution is too small? We first instrument the
full real optimization trajectory, then change one factor:its detached scale.

Selective prediction with an auxiliary prediction objective is established in
[SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html). Balancing
multi-task gradient magnitudes and training rates is established in
[GradNorm](https://proceedings.mlr.press/v80/chen18a.html). Regression routing
among experts is studied in
[Regression with Multi-Expert Deferral](https://proceedings.mlr.press/v235/mao24d.html).
Our simplified capped norm ratio is not a reproduction of full GradNorm;
none of these references establishes the selected positive-harm/reference
risk guarantee required here. We claim neither selective learning nor
gradient normalization as new.

## Controlled intervention

Let A denote the fixed five-moment/ranking/upper-score anchor and U the
current-admission auxiliary. Each batch optimizes `A + lambda U`, with
`lambda=min(1e6,||grad A||/||grad U||)` when the norms are defined and the
auxiliary norm exceeds1e-12; otherwise lambda is zero. Lambda is detached.
All other inputs, trajectories, model parameters, draws, optimization duration,
decoder and original2% all/easy risk criteria remain unchanged.

The matrix comprises18 fresh heads (six contexts,three seeds),36 verified
frozen controls,2,000 updates each,one exact complete refit and a real interrupted
pilot. Full-model endpoints were frozen before the72-group outcome readout.
Both cached controls reproduce their predictions, actions and complete metrics.
A separate process repeats the complete readout exactly.

## Statistical target and evidence role

The primary contrast is balanced-minus-unit full-policy ADE utility, with
known-label and wholly-unknown completion-lower versions. Localities, not
overlapping windows, are the bootstrap units:3,000 draws after context/seed
averaging. Twelve localities and74 recordings generate444 reused risk views.
All observations are exposed SOURCE fitting/resubstitution; the intervals are
conditional exploratory intervals, not independent generalization evidence.

ADE uses available future positions; FDE needs the final requested position.
The whole-roster FDE is undefined where a required view lacks endpoints.
Completion sensitivity concerns wholly unknown rows, not every missing
interior position. Count-matched utility contrasts cannot certify safety;
empty comparison support is not estimable, not a safe zero.

## Findings

The original auxiliary activates on93.81% of updates but its median norm is
0.003632% of the anchor. The repair changes this operating regime:its positive
scale snapshots have median achieved norm ratio0.132318, while28,464/36,000
updates have zero scale and6,163 hit the cap. These conditional and original
all-snapshot summaries are not directly paired populations.

Known ADE gain falls from1.904562% to0.388168%. The primary difference is
-1.516394pp,95%CI[-2.869731,-0.491213]; completion-lower difference is
-2.101592pp,CI[-3.424839,-1.044353]. Risk-supported views rise46/444 to201/444,
but31 known easy violations and72 undefined easy views remain. The repair
reduces interventions by90.28% rather than recovering a safe high-utility policy.

A secondary same-head physical-only restriction gives a small actual-neural-use
completion-lower increment+0.030673pp,CI[+0.003886,+0.079485]. It does not
replace the failed primary or imply an advantage from the new objective.

## Limitations and consequence

This result rejects one registered capped norm-ratio repair, not all
gradient-balancing methods, feature sufficiency or the possibility of a useful
scene-joint policy. Whole-record label-aware bank opportunity remains positive
but cannot prove inference identifiability or current-frame joint feasibility.
No independent role is opened and no policy is deployed from this experiment.
The data remain detector-silver top-down pixel/raw-frame trajectories, not
metric, seconds-level, human gold, true3D or foundation-model evidence.
