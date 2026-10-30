# One-factor selected-query loss-unit experiment

## Material Passport

The original36 paired critics are fully trained/frozen; their72-group policy
readout continues unchanged. This design follows real gradient telemetry and
the separately verified72-group training-label scale/capacity diagnostic,not
partial new policy scores. Commit before these new fits and outcomes. All
material remains exposed SOURCE fitting,resubstitution and diagnosis.

## Hypothesis and one changed factor

The original query loss divides a selected-query signed-score error by RMS
derived from ALL candidate actions. The measured scale is much larger than
the actual selected-query target,while its gradient is typically below0.1%
of the all-action loss. Replace only that query-score denominator with the
source-balanced RMS of its actual supervised quantity. Do not choose a global
lambda,confidence threshold,checkpoint or new feature from outcome scores.

For each fitting source,collect nonempty original selected queries whose
selected rows all have known labels. For each channel,compute the squared
signed target after dividing raw query mass by summed causal row scales.
Average these squared targets within source,average sources with support,
take the square root and floor at1e-6. Report unsupported sources rather than
invent zero labels. All sources/rows still enter training draws and inference.
This normalizer must match the verified capacity diagnostic at1e-12 relative
tolerance. Future labels set fitting loss units only; they never enter input
features or per-row inference.

Both loss weights stay0.5. Pointwise loss,architecture30,726 parameters,
initialization,step budget2,000,source/query sampler,causal normalization,
frozen benefit/reference/easy-reference values,all seven forecasts and
decoding remain exactly the parent design. The old query arithmetic is
recomputed and required to equal the parent's term bit-for-bit on actual
batches before changing its denominator. The new trace also retains the
original-unit query score,so unlike raw total losses can be compared properly.

## Full paired design

Cached_verified controls:36 original all-action-unit risk critics.
Fresh candidates:36 selected-query-unit critics,covering row-local/selected-set
context,six original contexts and cost-head seeds17/29/43. Primary training
updates72,000. First context/seed17 heads each receive a separate2,000-step
exact refit. The same single forecaster seed43 is frozen throughout.

Real600-update pilot:for each arm,reproduce the cached old100-step pilot via
the unchanged-unit wrapper;then compare50+50 resumed candidate updates against
a direct100-step fit. Require exact old scientific state except identity/time,
exact candidate recovery,and identical initial parameters/draws/input hashes.
Then complete all36 candidates and freeze before their policy evaluation.

## Prespecified outcome comparisons

Primary:new-units selected-set joint minus old-units selected-set joint full
paired wholly-unknown completion-lower utility. Also report known utility,
row-local normalization effect,and context comparisons within both unit modes.
No uncomputed difference-of-differences interval or special interaction claim.

Before new policy evaluation,the original critic72-group readout must complete
its separate full replay. Freshly infer/hash-check old critics and rescore their
immutable joint choices as cached_verified controls;do not reoptimize their
choices and claim they are fresh. New critics use the same decoder and per-
query common-count controls as the parent. Same-count comparisons are utility
only;empty is not estimable,not certified safe. Keep the exact original2%
complete-recording all/easy positive-harm and easy-degradation requirements.

Fitting advancement requires a positive primary lower CI,positive lower
utility versus floor,complete original recording-risk support,matched design
and exact control checks,and reported main-query optimality. Context-specific
effects remain secondary and cannot be claimed if their own comparison fails.
Even a fitting pass is not independent/deployment/submission readiness.

Retain all72 groups,unknown outcomes,hard/easy subsets,tails,regret,available-
label ADE,FDE availability and predicted-path geometry proxies. Use3,000 paired
locality bootstrap draws after repeated context/head-seed averaging over12
exposed localities. These are nominal development intervals,not independent
or adaptive-search-adjusted guarantees. The geometry diagnostic shows finite-
label target constraints;it does not prove conditional-mean misspecification
or that changing units will fix a deployed policy.

## Execution and limits

Native arm64 CPU4/interop1/workers0,checkpoints every100updates,heartbeat50,
explicit resume,12h healthy-run allowance,10GiB reserve and256MiB private cap.
Existing policies/data/progress are not overwritten. Original/new checkpoints
remain private. Publish only code/config/receipts/reports/light metric traces.

Task:obs8/pred12 at raw stride12,not supplemental raw-frame t+50. No newly
trained trajectory dynamics,metric/seconds,human-gold,true3D or foundation
claim. Independent roles,Stage5C execution and SMC remain closed.
