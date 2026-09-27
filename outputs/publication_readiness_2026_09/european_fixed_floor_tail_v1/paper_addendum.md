# Development Ablation: Tail-Weighted Fixed-Floor Risk Learning

## Material Passport

Fresh216 risk-head trainings with cached, provenance-verified forecasting and
floor models. Source-development ablation only; not an independent main test.

## Method

Let d be the frozen protected-damping prediction and n a frozen neural
prediction. Define R as d's observed ADE and H=max(ADE(n)-R,0). Analogous easy
moments include the producer-training easy-event indicator. Inference never
observes R,H or that future-derived event. Two matched neural estimators predict
four nonnegative moments from causal inputs. Harm outputs are bounded by the
candidate-rollout disagreement envelope. Fixed learned utility, movement and
fit-support guards are shared; both estimators screen predicted harm at2% of
predicted reference cost and otherwise retain d.

The treatment multiplies squared loss by4 above the fitting-only weighted90th
percentile of positive H, normalizing weights to mean1; reference moment loss
is unchanged. This estimates E[w(H)H|X]/E[w(H)|X], not E[H|X]. Increased caution
alone is therefore not evidence of conditional calibration.

The matched control uses the same ordinary-MSE scores to select the treatment's
exact number K of agents within each current recording/frame. It ranks by
maximum predicted all/easy signed budget excess and breaks ties by row ID.
Unknown-label rows participate. This prevents later frames or ground-truth
outcomes from allocating the intervention count.

## Result

| Policy | ADE gain over floor % | Intervention % | Selected harm % |
|---|---:|---:|---:|
| Frozen ridge |0.5233 [0.3790,0.6926]|5.6345|4.9104 [3.8936,5.9871]|
| Neural moment MSE |0.1544 [0.1154,0.1952]|7.8187|3.1126 [2.5204,3.6623]|
| Tail-weighted |0.0670 [0.0448,0.0925]|4.9228|undefined fixed roster|
| Count-matched MSE |0.0637 [0.0444,0.0836]|4.9228|undefined fixed roster|

The paired tail-versus-count-matched ADE difference is0.0033%
[-0.0025%,0.0121%]. Three empty held views make the prespecified harm-ratio
primary undefined.95 tail views exceed the2% risk budget despite preservation
of aggregate easy error. Thus this ablation does not support tail-weighted loss
as a reliable repair. Training loss decreases in all216 heads; that observation
does not change the downstream conclusion.

## Scope for a Future Manuscript

This is useful negative evidence separating coverage reduction from risk
ordering, with producer-chain exclusion and matched sampling. It is not yet a
standalone novel algorithm or a positive world-dynamics contribution. Relate it
to risk-coverage methods without claiming a reproduction of SelectiveNet or
the finite-sample guarantees of Learn then Test; see method_positioning.md.
Before any main claim: resolve source-conditional calibration, demonstrate
nontrivial neural gains at the risk budget, compare public matched baselines,
and evaluate once on appropriately reserved sources.

Detector silver, image-local raw-frame protocol; no metric, seconds, physical
safety, true3D or foundation claim. Stage5C/SMC remain off.
