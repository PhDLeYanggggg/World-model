# Current-decoder versus fixed-original query supervision

## Basis and role

The matched unit repair improves source-fitting known ADE,but its full robust
utility/risk requirement fails. Its complete score audit and separate replay
show better original-set MSE but residual harm underprediction on final choices.
Selection changes are observed;they are not proved to be the unique cause.
This experiment tests that explanation with one changed training factor.
All material remains exposed SOURCE fitting/resubstitution,not independent data.

## Exactly one changed factor

Controls:36 cached-verified selected-query-RMS critics from the preceding unit
experiment. Candidates:36 fresh row-local/selected-set critics in the same six
contexts and seeds17/29/43. Same30,726-parameter network,initial states,source/
query draws,all-action point loss,0.5/0.5 weights,2,000-update budget,original
selected-set context inputs,normalizers and all forecasts. Benefit/reference/
easy-reference remain frozen;only harm/easy-harm train.

Change ONLY the query loss's selected actions. At each actual update,detach the
current forward moments,restore exactly the original raw benefit/reference/
easy-reference values,and apply the same causal positive-gain independent rule
followed by the same joint decoder at that rule's count. The decoder receives
no targets,known masks,future endpoints or future/test goals. Use its chosen
actions for query-score supervision. Do not differentiate through the discrete
choice. This is moving-policy score regression,not an exact policy gradient,
SPO implementation or a decision-calibration theorem.

The model still conditions on ORIGINAL selected-set hidden context. Do not
introduce a circular inference-context fixed point. The query RMS remains the
preceding original-set fitting-target RMS,floored1e-6;it is not recomputed from
new choices or test labels. The point objective is exactly reused. Monitor both
current-set and frozen-original-set scores,with explicit units and support.

Current empty or selected-unknown queries stay in draws/context and contribute
no complete-query loss. Unknown rows never supervise point loss. Complete here
means available five-moment ADE-derived labels,not every future point/endpoint.
This change does not itself solve missing-label identifiability or reference bias.

## Runtime and actual pilot

Native arm64 CPU4/interop1/workers0. Checkpoint100,heartbeat50 plus wall-clock
selection heartbeat if a batch takes30s. Healthy jobs may run12h with10GiB free
reserve and256MiB private checkpoint cap. No multiprocessing/resource probing.
The training decoder uses the same2s/query limit and strict feasibility checks.
Any incomplete solver status REFUSES the training phase,not a silently variable
target. A numerical postcheck rejection retains the feasible independent anchor,
as in the fixed deployment decoder. Preserve/refine the design explicitly if a
real solver failure occurs;do not change settings under resume.

Real600-update pilot:perarm freshly reproduce the preceding fixed-target100-step
pilot (exact scientific state except identity/time),then compare50+50 resumed
current-target updates against direct100. Verify equal initialization/draws,
reference protection and no unknown supervision. Record real memory/runtime;
single-context full-cost extrapolation is a rough estimate,not measured completion.
If reasonable,train all36 candidates plus two separate exact2,000-step refits.
Final step is fixed;no outcome-selected checkpoint or threshold search.

## Complete evaluation

Require the preceding complete72-group policy replay before new policy scoring.
Use the unchanged seven forecasts,2% all/easy budgets,full cohort and recording
risk requirement. Re-infer/hash-check the36 fixed-target controls and rescore
their immutable choices as cached_verified,not freshly reoptimized. Verify all
144 fixed full-policy scores. Newly infer/optimize the current-target heads and
their row/set common-count controls. Equal-count comparisons are utility-only;
empty comparisons remain not estimable,not safe-zero risk.

Primary:current-query set-joint minus fixed-original-query set-joint full paired
wholly-unknown completion-lower utility. Also report known utility,row-control
effect,context contrasts,unknown support,hard/easy subsets,tails and path proxies.
Use3,000 paired locality bootstrap draws after repeated context/head-seed
averaging over12 exposed localities. These are nominal exploratory intervals,
not adaptive-search-adjusted or independent confirmation. One forecaster seed43
remains fixed;three cost-head seeds are not three forecast producers.

Fitting advancement requires positive primary lower CI,positive completion-lower
utility versus floor,complete original recording-risk support,paired controls
and main-query optimal-status/postcheck acceptance. No risk or missing-support
criterion is relaxed. A fitting pass alone cannot promote deployment/submission.
Require a separate full72-group replay. Publish code/config/reports/light metrics,
never raw packets,per-row data or checkpoints. Completed evidence is immutable;
genuine interrupted phases use explicit resume with identical scientific binding.

Task:obs8/pred12 at raw stride12,detector-silver/local-image coordinates.
No new trajectory forecaster,metric,seconds,human-gold,true3D or foundation claim.
Independent roles remain closed;Stage5C execution and SMC remain disabled.
