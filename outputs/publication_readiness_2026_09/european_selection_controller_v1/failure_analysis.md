# Failure Taxonomy: Complete SOURCE-To-Selection Controllers

## What Was Actually Tested

Eighteen real cost controllers fit eight producer-excluded SOURCE localities
per model. They use real past geometry/detector histories and fixed causal
rollouts, not evaluation metadata or hard best-action labels. Both arms share
29,253parameters, initialization, draws, coverage and loss. The model-selection
evaluation retains all38,102targets and the original2%record requirements.
No new trajectory model, threshold search or calibration-set exposure occurred.

## Measured Failures

1. **No reliable joint-allocation gain.** The registered matched-joint lower
   effect is+0.0016077pp [-0.0152463,+0.0168448]; its known component is+0.0137784pp
   [-0.0080937,+0.0344373]. Both intervals cross zero. This is not a substantive
   positive joint contribution. Free-count gain0.2173% does not rescue it.
2. **Original safety still fails.** Context matched-joint supports21/252repeated
   recording views, with157known-all and156known-easy violations,52undefined
   easy views and7.0157%worst known easy net degradation. Independent is also
   unsafe (23/252support,3.7724%worst easy). Missing outcomes or empty support
   do not explain away the observed positive-harm violations.
3. **More same-query context does not repair the critic.** Context-minus-row
   matched-joint known effect is-0.0012360pp, with an inconclusive known interval;
   completion lower is-0.2111303pp [-0.3462881,-0.0958706]. A pooled representation
   is not evidence of better selected-risk identification. The loss means fall
   from3.40376to0.26793(row)/0.30407(context), but these varying SOURCE minibatches
   neither certify generalization nor distinguish underfit from distribution shift.
4. **The controller captures little of raw forecast utility.** Raw history has
   known ADE gain11.6729%; context matched-joint has0.1733%. A neural-action bank
   has a small known increment over the physical bank,+0.05748pp [0.03197,0.07749],
   but its completion-lower interval includes zero. This is not new safe dynamics.
5. **Switching is not absent.** The causal-only audit finds19,496context switches
   across342,918repeated row views. Joint allocation changes the composition
   from12,060neural/7,436physical to11,259neural/8,237physical at the same count.
   It substitutes801neural actions with physical ones, not simply more neural
   predictions. These are repeated views, not independent new samples.
6. **Admission is selective but not reliably safe.** Context predicts positive
   neural gain on71,298rows;18,899pass both predicted risk guards. Another52,399
   gain-positive predictions fail a guard. Those are *predicted* gains, not
   observed beneficial opportunities. The audit reads no future labels, so it
   does not prove false rejections, actual calibration or irreducible uncertainty.

## What This Rules Out And Does Not Prove

This failure is not explained by no real Torch training, missing full-query
coverage, only one actual forecasting seed, a changed forecast bank, a zero-switch
policy or a successful program being mistaken for scientific success. Full
budgets and exact first complete refits finish. Every query's predicted budget
feasibility and matched count are checked during separate replay. Some solver
proposals revert to the feasible parent; those counts remain in every policy
receipt. Solver optimization is not claimed reexecuted by metric verification.

The results do not uniquely identify insufficient data, causal-feature
insufficiency, cost misspecification, calibration transport, selected-reference
bias or irreducible future uncertainty. They also do not prove that extending
training or adding another network will fix the issue. Pairwise composition
conflicts have not yet been optimized; this is a risk-pooling control only.

## Next Distinguishing Action

Audit this same frozen critic's all-eligible versus selected signed risk on
SOURCE and the exposed model-selection localities. Separate H underprediction,
R/easy-reference overprediction, false-low-risk admission and useful rejection.
Compare input shift and numerical units before changing capacity or thresholds.
The current causal action counts are the first executed part of that diagnosis.
Future source/selection comparisons must keep fitting and evaluation separated;
selection outcomes may guide method choice but are never an untouched final test.

Do not select favorable records/seeds, delete unknown targets, weaken2%, relabel
the known secondary as primary or open final confirmation to tune the repair.
Calibration/confirmation/DroneCrowd remain closed. No deployment or submission
promotion; no Stage5C/SMC. Pixel/raw-step detector-silver2.5D evidence only.
