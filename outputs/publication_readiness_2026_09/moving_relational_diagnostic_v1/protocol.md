# Past-Neighbor Relative-Motion Diagnostic

This additional diagnostic implements the existing interaction-consistency
requirement. Its definition is frozen before the complete parent forecast
readout. It changes no absolute ADE/FDE,primary,source role,training,2%risk,
policy selection or deployment gate. It cannot rescue a failed original gate.

On each admitted query,retain all past-complete target agents used by the
parent bank. Construct the undirected union of each target's eight nearest
current target neighbors,breaking distance ties by agent ID. No future
position or future-validity mask participates in edges. Queries with fewer
than two targets have no pair task; they are not imputed perfect predictions.

For each edge `(i,j)`,evaluate the error of the predicted relative future
displacement `r_i(t)-r_j(t)` against the labelled displacement difference.
Both agents must be observed at a future index to label that pair/time.
Report available-label pair ADE,endpoint pair FDE,complete/partial/wholly
unknown support,and the paired EqMotion-minus-PCT difference. Wholly unknown
pairs use the maximum disagreement between their two predicted relative paths
for a bound on the unobserved error difference,not fabricated pair locations.
Percentage-point contrasts use the fixed known CV pair-ADE mass; they are not
full12-step ground-truth gains or a population risk certificate.

Use the same frozen18models,three actual seeds,all parent row IDs and both
producer-excluded folds. Relative predictions must be aligned to their own
observed agent origins. CV and the producer-fitting-selected floor are causal.
Future labels are evaluation-only. No new model,teacher,normalization or goal
construction occurs. Wait for complete parent inference/full replay before
computing the actual diagnostic. Preserve the parent original24-group primary
as not estimable and label all23-supported-group outputs as secondary.

Within a source group,pairs are edge-weighted. Average producer/seed views
before3,000source-group bootstrap resamples,not resampling overlapping edges
as independent observations. Retain required empty or zero-reference groups
as not estimable rather than dropping them to obtain a positive interval.
Report per-group/per-seed support. Source components are not verified IID sites.

**Common-translation blind spot:** an arbitrarily wrong displacement shared by
all predicted agents cancels exactly in this metric. Perfect pair error can
coexist with catastrophic absolute position error. Always report the parent
absolute errors beside it. This is an image-plane relative-motion proxy,not
published multimodal JADE/JFDE,a collision measure,metric ground geometry or
physical safety. Pixel/native-step only; no seconds,true3D,foundation,
Stage5Cexecution orSMCclaim. See the
[primary-source positioning note](../selected_ratio_prior_update_20261103.md).
