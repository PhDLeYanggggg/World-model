# Single-Factor Auxiliary Prior Repair

## Question and Scope
The preceding exact training reconstruction found an auxiliary initialized with
easy-membership probability (median about28%) although its cap-event target is
much rarer (full median5.1%,motion-only1.1%). This is a post-hoc hypothesis,
not a proven cause. Does correcting only that intercept repair expected-cost
learning and transport to the existing source-held development localities?

This is an engineering mechanism experiment,not an architectural novelty or
an independent test. Historical development exposure cannot be erased.

## Fixed Intervention
Two new arms:cap_aux and locality-shuffled_aux. Initialize only membership.bias
to logit(q),where q is the original fitting-weighted cap-event prevalence,
conditioned on finite fitting event labels and positive envelopes. Apply the
original1e-6 logit clipping. Both arms receive the same fitting prior.
No held outcome estimates the prior. No independent-role data is used.

Keep original383 features,GELU64 shared network,four-cost output structure,
cost/shared initialization,labels,site-balanced batch draws,loss normalization,
auxiliary coefficient1,AdamW settings and2000 updates. Preserve unknown-label
masking and all-known cost support,including zero-envelope cost rows.
New implementation with inherited_easy must exactly reproduce the frozen loop
in scoped tests. Original sealed code is not changed.

144 views:6 producer/controller assignments,3 seeds17/29/43,2 full/motion
families,4 outer localities. Two heads/view,288 heads,576000 updates.
Reuse hash-verified original cost_only,true/shuffled controls;they were already
reconstructed exactly in the preceding study. Do not call reused controls fresh.
Pilot:first registered true head200 updates,then resume to2000. No early stop,
model search or checkpoint choice. Every final prediction must be committed
before the source-held cost readout. Reference cost columns remain frozen.

## Prespecified Readout and Gates
Primary:expected easy-harm MSE gain on positive-envelope source-held rows.
Contrasts:repaired_true versus cost_only,old_true,and repaired_shuffled;
secondary repaired_shuffled versus old_shuffled and old_true versus cost_only.
Guards:top10 harm-mass capture,absolute log harm-coverage error,and all-row
all-harm MSE. Preserve parent metrics,edges fitted only on fitting scores,
row alignment,known support and original outer-target hashes.

Average3 seeds within each locality,then3000 paired resamples of4 localities
per assignment (seed71429). Report all6 assignments and both families,including
unsupported cells. No multiplicity correction or independent-confirmation claim.
Fitting results are secondary diagnostics using dependent fitting views.

Advance only if all6 full primary intervals are strictly positive against both
cost_only and old_true,none of the required guards has a negative or missing
interval,and all6 true-versus-matched-shuffled primary intervals are positive.
This strict development gate does not authorize deployment even if it passes.
Event classification,lower BCE,or improvement over shuffled alone cannot pass.

## Execution and Boundaries
Local native arm64 .venv-pytorch,CPU4/interop1/workers0. CREATE checked read-only;
existing jobs untouched. Checkpoint/heartbeat every200 updates,atomic writes,
exclusive lock,resume identity checks;preserve10GiB disk. Do not stop for slowness.
Checkpoint and row-level outputs stay private and ignored by Git.

Result sources:fresh_run for new training/readout;cached_verified for sealed
controls/features/forecasts;not_run for new trajectory/policy training or
independent selection,reserved calibration,confirmation. Shared containers are
not filesystem-blind;the claim is no held outcomes in fitting,not blind storage.

Main protocol is obs8/pred12 native annotation steps and detector image pixels.
No metric/seconds,physical-safety,human-gold,true3D or foundation claims.
Deployment unchanged. Stage5C and SMC remain prohibited.
