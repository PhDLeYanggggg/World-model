# Coherent Costs Trained On Fixed Versus Own Decisions

## Material Passport

Prospective control following the completed72-group zero-step projection.
That postprocessor removes cost contradictions,but worsens known and full
completion-lower utility and fails the original recording-risk requirements.
Selected-action H/H_easy remain underestimated on observed SOURCE labels.
Those descriptive errors motivate this test;they do not establish its outcome.

All present data remain exposed EuropeanSquares SOURCE fitting/resubstitution.
No independent model-selection,calibration or confirmation labels are opened.
Seven trajectories,462causal fields,normalizers,targets,availability and the
original149,510targets/74recordings/12localities/72groups remain unchanged.
Three cost seeds17/29/43share forecaster43,not three full forecast pipelines.

## One Controlled Factor

Both arms warm-start the same preceding learned-reference cost weights for each
context/seed. Both have31,119total and active parameters,identical architecture,
initial state,AdamW settings,cyclic query draws and complete-update budgets.
The coherent decoder is shared:in native units,average R/R_easy over causally
available candidates per agent,cap R_easy at R,and B at shared R. Preserve H/H_easy.
The same transform is differentiable during training,not just applied afterward.

The prior native reconstruction arithmetic is retained before projection.
Torch/Numpy reduction order may differ by rounding;record the initial maximum
causal-scale-normalized difference and require it below1e-10. This is not claimed
as bitwise reproduction of the preceding Numpy postprocessor. Paired arm
initial weights and their own predictions must be bitwise identical.

Fixed arm:query-level signed-risk residuals use frozen causal `initial` choices.
Own arm:use the current critic's detached native costs with the unchanged
independent selector and query joint allocator at its own independent count.
The original two predicted2%constraints,positive-gain eligibility,2s/query limit,
and feasible-independent fallback remain. No future label/label-validity mask
can enter these choices. No new action,no-op anchor or enlarged risk budget.
The model's selected-set context input itself remains the same frozen causal
initial set in both arms;only the aggregate loss's selected actions change.

## Objective And Unknown Outcomes

Both retain the same all-legal-known pointwise H/H_easy and benefit losses,
reference MSE,and frozen SOURCE RMS normalization as the preceding refit.
Only its query-level residual is substituted. For selected actions C and
moments(H,R)or(H_easy,R_easy),compute:

    e_q = sum_C[(H_pred-.02R_pred)-(H_true-.02R_true)] / sum_C(causal_scale)
    L_query = mean_q 1[nonempty_C and all_selected_labels_known] (e_q/RMS_q)^2

The common total is the preceding total with half the old query term replaced
by half this query term. Fixed choices must reproduce the preceding loss's
arithmetic exactly. Both arms keep pointwise supervision on every legal known
candidate,including candidates that are not chosen.

An empty selection contributes no query target and is counted separately.
A selected query containing unknown outcomes has no supervised query residual;
its known rows still receive pointwise supervision. Unknown outcomes are never
supervised zero. Causal unknown rows remain in context,selection and sampling.
This is a stop-gradient choice-conditioned regression control,not a gradient
through an integer solver or an implementation of a published consistency loss.

## Registered Primary And Advancement

Primary:own-choice joint minus fixed-choice joint FULL completion-lower ADE
utility,lower95%CI>0. Also require improvement over the frozen projected learned
joint policy with lowerCI>0,own-floor lowerCI>0,and every original recording's
all/easy positive-harm/reference and net-easy requirements. The2%contract is fixed.
The fixed arm is not an alternative primary if the own arm fails.

Retain independent,joint and common-count controls for each arm,plus the exact
preceding projected learned joint policy and floor. Report full and post-hoc
trimmed comparisons;empty required comparisons stay not_estimable,never safe0.
Joint-versus-independent is a secondary component contrast,not a safety claim.

Report known ADE,full-roster FDE availability,completion bounds,all five cost
errors,selected bias,per-record risks,easy tails,hard/easy subsets,interventions,
unknown selections,pointwise label-aware opportunity,pair contraction and solver
fallbacks. Witnesses are diagnostics,never model utility or inference inputs.
Use3,000paired locality bootstrap draws,seed20261101,after averaging repeated
context/cost-seed views. These are conditional SOURCE fitting intervals.
No endpoint substitution,fabricated interior labels or seconds/metric claims.

## Actual Execution And Recovery

First run a real400-update pilot with interrupted/direct100-update fits in both
arms and an18-query native allocation recovery probe. No policy outcomes are
scored in the pilot. Measure elapsed time and peak RSS before deciding venue.
The previous complete local readout took30.25minutes and11.20GB;local48GBmemory
is adequate for preparation/inference,but the new solver-in-training cost must
be measured rather than assumed. No unnecessary CREATE job is launched.

Formal training:minimum2,000updates and at least four full cyclic passes over
the largest source's queries,with16complete queries/batch,equal source draws.
The unchanged roster implies208,320primary optimizer updates across36heads.
Repeat both full-budget first-context/seed17fits from scratch as recovery
verification,adding14,336updates. Compare model,optimizer,RNG,sampler,loss traces
and decision statistics exactly;exclude only measured wall-time fields.
Timed-solver choice divergence is an actual reproducibility failure,not hidden.

Native arm64 .venv-pytorch,CPU4/interop1/workers0,no resource probing or worker
processes. Checkpoint every50updates;trace every50updates and live allocation
heartbeat at least once per minute of active allocation. Query readout saves
every250queries. Explicit same-definition resume and process lock,12hphase
boundary,10GiBdisk reserve and1GiBown private checkpoint cap. No speed-based
data reduction. Interrupted incomplete phases are resumed,not called complete.

Final verification re-infers costs and scores all frozen actions,not timed MILP
reoptimization. Public scalar reports must rebuild without private input reads.
No raw data,features,caches,weights,images/videos,third-party PDFs or .venv go toGit.

## Method Positioning And Limits

Existing multi-expert regression deferral and feasibility-aware decision-focused
learning already cover cost-sensitive selection and learning constraints.
This fixed/own-choice residual comparison is not claimed as a new theorem or a
literal implementation of either published loss. Strong adapted baselines and
independent full-pipeline evidence are still required for a submission claim.

EuropeanSquares oblique-camera detector-silver/local-image,obs8/pred12/rawstride12.
Not SDDt50,meters,seconds,human gold,true3D or foundation. No forecaster,JEPA,
Transformer,residual or latent generator is trained. Stage5C and SMC remain off.
SOURCE fitting improvement alone cannot authorize deployment or submission.
