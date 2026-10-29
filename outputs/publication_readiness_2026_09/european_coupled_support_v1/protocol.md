# Joint Risk Moments and Recording-Supported Blending

## Material Passport

Registered before real fitting. Follow-up to local-cost v2, not a new neural
forecaster or independent evaluation. Original TRAIN inner-fit, obs8/pred12,
stride12 raw frames, pixel/local detector-silver. Keep the 2% positive selected
harm/reference budget; this is not net easy ADE degradation. Inner-held is
discarded before forward; outer dev and independent roles remain unopened.
Query knownness scores only. Upstream exposure prevents whole-pipeline OOF.

## Observed Failure and Hypotheses

Previous local cost fitting reduced uniform-neighbor easy MSE by81.30%, but
remained43.00% worse than the original head;0/18 pairs beat both controls.
It produced9,019 EH>H occurrences. Row effective support reached1.00013
despite32neighbors. Those observations motivate two controlled changes:
joint constraints and reference-record-excluded estimation of blend strength.
Coupling alone cannot be presumed sufficient: weighted harm was coherent but
also failed the original head. Shrinking toward the original is not automatically
conditional information; mean/shuffled blend controls must remain visible.

## Frozen Design

Retain all72 views (6contexts x3existing head seeds x4sources),62902 repeated
queries,380causal features,32neighbors,full causal-bank zscore excluding query
recording,std floor1e-6,clip8,brute Euclidean,n_jobs1. Known selected donors only
supervise. Query's source-qualified entire recording excluded from donor labels
and normalizer. Unknown queries stay. Parent origin/local-cost outputs, donor
and normalization hashes must match exactly. No feature/k/threshold search.

## Joint Quadratic Solution

For donor i, E is causal envelope, S fixed TRAIN source scale, and N_source
counts selected known donors after excluding the query recording.

    t_i = ([H_i,EH_i] + .02*([Rhat_i,ERhat_i]-[R_i,ER_i]))/E_i
    w_i = (E_i/S_i)^2/N_source(i)
    m = sum(w_i*t_i)/sum(w_i)
    cap_q = clip((E_q-Bhat_q)/E_q,0,1)

Minimize the same equal-curvature two-channel quadratic subject to
0 <= f_EH <= f_H <= cap_q. If m_EH>m_H, pool both unbounded means; then clamp
to [0,cap_q]. This is the exact joint optimum, not clipping an already fitted
independent output. Output E_q*f and retain frozen B/R/ER. All joint arms must
satisfy EH<=H and B+H<=E (numerical audit tolerance1e-7).

## Donor-Only Reliability Fit

Within each query's SAME32 neighbors, hold each donor's entire recording out
of that local fit. Do not refill neighbors. For a remaining donor j, recompute
N_source by subtracting the full held recording's selected donor count from
the already query-excluded full reference bank. Refit the joint quadratic;
use the held donor's causal cap. All32 from one recording means no nested
reference support, so alpha=0 (original risk estimate, NOT safe zero).

Let e_i be the original normalized signed-risk error on a held donor and d_i
the difference between its nested joint and original harm predictions /S_i.
Give each donor recording equal total weight, w_eval=1/local_record_count,
renormalized over32 donors. Learn both channels jointly:

    alpha = clip(-sum(w_eval*e*d)/sum(w_eval*d*d),0,1)
    H_query = Hhat_original + alpha*(Hhat_joint-Hhat_original)

Denominator<=1e-24 or missing nested support gives alpha0. No query labels,
future endpoints or future knownness in this calculation. This nested loss is
training on donor labels, not calibrated generalization. Original normalization,
neighbor membership and upstream models remain exposed; NOT nested full-pipeline
cross-validation. Convex blending preserves the joint bounds.

## Six Arms

1. origin: frozen reference head.
2. independent_cost: exact prior local_signed32.
3. coupled_cost: joint constraint only.
4. record_mean_blend: same mean alpha for all queries in the same recording.
5. shuffled_alpha_blend: within-query-record alpha permutation, seed20261029.
6. nested_support_blend: PRIMARY donor-trained, query-specific alpha.

Mean/shuffle controls preserve alpha mass within recording, not risk or actual
intervention count. They require batch causal queries, not future labels.
Fewer than32 external donors retains the original in joint/blend arms; local
support separately fails the screen. Independent-cost remains undefined there.

## Evaluation and Non-Promotion

Primary unchanged: equal-view normalized signed-risk MSE, not ADE/FDE.
Preserve previous five conditions and strengthen against independent cost:
full local support; easy MSE below original, uniform and independent; all MSE
no worse than uniform and independent;>=12/18 pairs beat original and independent
easy MSE; joint moments valid. All eight conditions required. Record conditional
assignment screen separately: primary easy MSE below mean AND shuffled alpha.
No switching primary or relaxing screen after seeing results. Even a pass does
not authorize deployment, held evaluation, or new scientific claims.

Report all source/context/head-seed slices, event/non-event harm, false-safe
counts, row and recording ESS (weight concentration, not independent sample
size), alpha quantiles/zero/one, nested losses and counts. Prespecified
descriptive slices: one/multiple donor recordings,recordESS<2/>=2,alpha0/>0;
cross-tabulate old EH>H with rowESS<2 and recordingESS<2. No slice selection.
No CI/p-value from exposed fitting repeats. Head seeds are not new forecasters.

## Execution

Register and commit code/tests/protocol before pilot. Pilot includes diagnostic
writer and exact controls; then72 fresh fits, full exact replay, separate
receipt arithmetic. Local arm64 CPU4/interop1/workers0, read-only authorized
TRAIN packets, no new CREATE scheduler job. Atomic per-view checkpoints,
heartbeat/process lock/resume,10GiB disk reserve,12h bound. No full data cache.
No metric,seconds,human-gold,physical-safety,true3D,foundation or submission-ready
claim. Stage5C and SMC remain off. All failures retained.
