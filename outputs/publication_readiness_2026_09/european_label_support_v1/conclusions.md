# Label Support Is Not the Whole Failure Mechanism

## What Was Actually Computed

This is a fresh source-development diagnostic, not new training. All72 original
forest cost heads were evaluated with unchanged features, splits, predictions,
actions and the2% selected-positive-harm budget. Past-quality arrays and model
weights are cached_verified. Future masks/coordinates were freshly reconstructed
from the admitted raw tracker records. Future quality and error labels are used
only for this offline analysis, never as inference features or filtering rules.

The raw check covers318,969 rows,163 recordings and3,221,201 valid future
coordinate labels reconstructed from raw box centers.
All requested future masks and coordinates match by recording/tracker/frame.
This rules out a detected packing mismatch; it does **not** certify tracker
identity, detector correctness, or human-gold labels.

The72 model views contain596,988 repeated row occurrences, corresponding to
99,498 distinct validation rows,14,848 recording-queries,19,824 scoped tracker
IDs and58 recordings. There are95,455 selected occurrences, but only24,842
distinct selected rows.12,733 harmful occurrences correspond to3,997 rows harmed
by at least one frozen head. Repeated heads/windows are not independent samples.

## Main Result

**Selected harm is not confined to incomplete future supervision.**
All12-label rows account for73.11% of selected harm mass after averaging within
locality and then equally across the11 localities with defined harm share.
The nominal3,000-draw locality-bootstrap interval is[61.05%,84.16%]. Locality082
has no defined selected-harm share and is not silently assigned zero.

The pooled occurrence-weighted share is66.49%; this is a different denominator,
not a conflicting result.8,285 of12,733 harmful occurrences (65.07%) have all12
future labels. Missing-label filtering alone cannot remove these known failures.
Complete detector labels can nevertheless still be inaccurate.

| Future labels present | Selected occurrences | Known harmful | Harm mass | Pooled easy harm/reference |
|---|---:|---:|---:|---:|
| 0 |918|unknown|unknown|undefined|
|1-3|2,005|756|32.2663|0.8526%|
|4-11|21,553|3,692|565.9934|0.7053%|
|12|70,979|8,285|1,186.8617|0.5625%|

These pooled risk ratios do not supersede per-group failures or unknown-outcome
bounds. The complete-label stratum still has four violations among 40 defined
group ratios; 32 are undefined. Its worst easy harm/reference ratio is 7.6756%.
This secondary stratified readout is not a new filtered deployment policy.
918 selected unknown occurrences retain1,520.3833 units of disagreement
envelope, not zero harm. No safety/deployment gate is promoted from this table.
Harm masses are sums of image-coordinate prediction-error differences across
repeated views, not physical harm or independent-patient/agent risk.

## Temporal Sensitivity

Removing one available time point changes the harmful/nonharmful sign in
1,967/12,459 defined harmful occurrences (15.79%).274 singleton-label harmful
occurrences have undefined leave-one-out sensitivity, not demonstrated stability.
Within complete12-label harmful rows, the fraction is1,106/8,285 (13.35%).

The first and second six-step halves have opposing signed error in4,738/11,595
defined harmful occurrences (40.86%); among complete-label rows it is3,330/8,285
(40.19%). This supports investigating temporal benefit/harm structure. It does
not establish annotation noise: a legitimate trajectory can change which
forecast is better over time. No official error was recomputed after omitting
time points and no new target has been selected from these outcomes.

## Quality Associations

The following values are harmful-minus-nonharmful selected means, matched within
the same recording/frame. Repeated heads/controllers are averaged within each
locality before bootstrap. Only eight localities have defined paired contrasts;
the unmatched descriptive contrasts have nine. Intervals are nominal, not
multiple-testing corrected and not independent confirmation.

| Proxy | Paired mean difference | Nominal95% locality interval | Input status |
|---|---:|---|---|
| Past line-fit residual/current width |+0.00985|[+0.00425,+0.01577]|past-only|
| Past raw-frame presence fraction |-0.00533|[-0.01148,-0.00065]|past-only|
| Past detector confidence |-0.01727|[-0.03559,+0.00012]|past-only|
| Past last-step/OLS disagreement |+0.00085|[-0.00143,+0.00336]|past-only|
| Past width-range/current width |+0.00546|[-0.02231,+0.02456]|past-only|
| Past reversal fraction |-0.00247|[-0.02908,+0.02713]|past-only|
| Partial nearest-neighbor count |+0.16945|[-0.04947,+0.50188]|past-only|
| Future detector confidence |-0.02079|[-0.03723,-0.00498]|offline only|
| Future max width change/current width |+0.04673|[+0.00864,+0.08313]|offline only|
| Future class-ID-change indicator |0|[0,0]|offline only; not identity proof|

The past confidence interval overlaps zero, so there is no supported claim that
confidence alone solves selection. The partial-neighbor signal loses support
after within-query matching, consistent with retaining the earlier negative
partial-neighbor retraining result. Future confidence and box changes may reflect
occlusion, perspective, real motion or tracking error; these causes are not
identified here. They must not be turned into a deploy-time quality filter.

## Decision

Do not relabel trajectories, drop incomplete/easy/harmful rows, retrain on future
confidence, or call this a safe selector. Do not repeat global calibration,
relative-leaf scaling or the already-negative partial-neighbor-only repair.

The next falsifiable step is a train-only **past-observation-quality auxiliary**
with the entire fixed past-proxy bundle, compared with the unchanged parent at
matched intervention counts. This tests predictive value rather than declaring
noise from correlations. It has not been trained in this diagnostic. A temporal
gain/harm target is a separate subsequent hypothesis, not combined here to make
attribution impossible. An externally adjudicated identity/label subset remains
missing; numeric tracker consistency cannot replace it.

The model is still a2.5D trajectory/world-state research candidate, not true3D,
foundation, independently confirmed or submission-ready. Task:obs8/pred12,
raw stride12, image-local detector-silver. No metric, seconds or physical-safety
claim. Independent roles remain closed; Stage5C/SMC are off.
