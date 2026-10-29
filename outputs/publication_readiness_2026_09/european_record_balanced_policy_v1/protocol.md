# Record-Balanced Constraint Experiment

29 October 2026. This is a TRAIN-only continuation of the cost-sensitive
fitting repair, not another independent test or a deployment update.

## Hypothesis

The cost-sensitive loss recovers81.55%of eligible positive benefit but passes
only84/444record/head views. Of346known all-risk violations,343are in two
multi-recording sources. No all-risk dual reaches cap100. This motivates a
finite-time weighting hypothesis: the current primal penalty scales every
record's constraint by its utility weight, which depends on its row count
within its source. Small recordings may receive too little optimization weight
relative to the requirement that every recording pass.

This observation does not prove weighting is the cause. Limited features,
shared-model capacity and incomplete convergence are alternatives. The new
experiment changes only constraint weighting. If it fails, retain the failure.

## One Changed Quantity

Let w_j be the existing positive record utility weight, with sum_j w_j=1,
and J the number of fitting records for the context. Keep the utility objective,
data, architecture, source preprocessing, optimizer, dual update, initialization,
temperature schedule and final iteration unchanged. Set constraint weight
q_j=1/J instead of q_j=w_j. Inside the already w_j-weighted row loss use:

`c_ij = -gain_lower_ij + (q_j/w_j) * sum_k dual_jk * risk_ijk/scale_jk`.

The cost-sensitive logistic surrogate remains abs(c)*softplus(sign(c)*z/T).
This reallocates a total constraint weight of1; it does not uniformly increase
the penalty. A large recording's penalty can decrease. The dual player still
uses exact per-record hard-action excess, lr1, cap100. Neither budget signs nor
the2%requirements change. Record IDs are TRAIN constraint groups, not new
inference features. No new threshold, label gate or oracle action is an input.

The established constrained/cost-sensitive framework is not a novelty claim:
[Agarwal et al., ICML2018](https://proceedings.mlr.press/v80/agarwal18a/agarwal18a.pdf),
[Cotter et al., ICML2019](https://proceedings.mlr.press/v97/cotter19b/cotter19b.pdf).
This nonconvex, finite-iterate heuristic inherits no best-response, feasibility,
or independent-data generalization theorem from those papers.

## Fixed Comparison

18new fits: six frozen forecast/controller contexts x head seeds17/29/43,
2,000updates each,385->32->1policy,AdamWlr.001,weight_decay.0001,gradientclip5,
temperature1to.25. The frozen cost_sensitive finals are the primary control;
origin is secondary. All data, models and parent metrics are hash-verified.
No extra repeated control training except a20update engineering replay.
All18finals are committed before the72fitting readouts. No best seed,
checkpoint or threshold selection. Evaluate every original fitting recording.
New inner-held scores and independent selection/calibration/confirmation are
not run. The original held subset is historically exposed, not independent.

Primary evidence: full-policy per-record all/easy harm budgets and net easy
degradation, utility completion bounds, known benefit/harm retention, source
and head-seed slices. Count-matched comparisons are utility only; empty risk
is not_estimable, not zero or pass. No TRAIN bootstrap masquerading as test CI.
All72views and444record/head views remain; underlying74unique recordings and
148record/context views are the correct repeated-sample accounting.

The locality082/51 candidate/support obstruction remains in two contexts. This
experiment cannot certify the complete roster using its fixed candidate set.
Any improvement elsewhere is a fitting mechanism result, not permission to
drop that recording or open confirmation. Candidate/support repair is separate.

## Runtime and Recovery

Local native arm64 .venv-pytorch,CPU4/interop1,workers0. The60update real pilot
checks balanced10+10resume versus20direct and neutral20updates against the
frozen parent's20update pilot. Pilot predictive performance selects nothing.
Checkpoint100updates,heartbeat50,process lock,explicit resume,128MiBcheckpoint
cap,10GiBfree reserve,12h invocation boundary. Existing parent assets are read
only. A slow healthy run continues; a restart requires terminal-state evidence.

Fresh_run:new neural policy training and fitting readout. Cached_verified:
forecasts,raw-source-derived feature packets,moments and controls. Not_run:
new trajectory training,held/independent evaluation,deployment. No Stage5C/SMC,
metric/seconds,true3D,foundation or submission-ready claim.

```sh
.venv-pytorch/bin/python -m scripts.run_m3w_record_balanced_policy register
.venv-pytorch/bin/python -m scripts.run_m3w_record_balanced_policy pilot
.venv-pytorch/bin/python -m scripts.run_m3w_record_balanced_policy train
.venv-pytorch/bin/python -m scripts.run_m3w_record_balanced_policy evaluate
.venv-pytorch/bin/python -m scripts.run_m3w_record_balanced_policy check
```

Only an interrupted phase uses --resume. Data,caches and weights remain private.
