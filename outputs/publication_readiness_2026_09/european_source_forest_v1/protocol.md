# European Source Nonlinear Cost Regression Control

Status: preregistered development diagnostic, not independent confirmation.
Question: do conventional nonlinear cost regressors learn the frozen five causal
cost targets more reliably than the existing neural moment head?

## Fixed Assets and Scope

Reuse all 72 verified neural fitting partitions (18 contexts, three seeds
17/29/43, four fitting localities per context). Recompute the same whole-recording
optimization/validation split and optimization-only preprocessing and require
exact agreement. Twelve previously exposed development localities, 216 directed
transfer views. Independent selection, calibration and confirmation remain closed.
No test relabeling, future-label availability in eligibility, threshold sweep,
new goals, data split, metric or risk-budget change.

Obs8/pred12, stride12 raw frames; image-local detector-silver, not metric,
seconds, human gold, true3D, foundation, physical safety or a deployment claim.
Stage5C execution and SMC remain off.

## Estimator and Objective

One fixed ExtraTrees regressor per source fit:128 trees, depth12, min leaf64,
max features1/3, no bootstrap, seed inherited, four threads, no multiprocessing.
Checkpoint every16 trees; no checkpoint/model/hyperparameter selection.
Pilot16 trees resumes into the same128-tree fit, not a small substitute.

380 clipped standardized causal features plus log1p(causal disagreement /
optimization reference scale). The original neural decoder receives that same
envelope separately; 381 tree inputs do not introduce new future information.
Same five targets: benefit, positive harm, floor error, easy floor error, easy
positive harm. Entirely unknown outcomes are excluded from fitting; native partial
labels are retained. Optimization-only source/query weights average agents within
queries and queries within the single fitting locality. Trees fit all weighted
known rows, not the finite neural query draws; this is not equal compute.

Eight outputs are the five moments plus gain, harm-.02*reference and
easy_harm-.02*easy_reference. Divide by the frozen training reference scale and
eight RMS scales. Multiply the five moments by sqrt(.8) and the three signed
scores by sqrt(4/3). Mean8 squared error then equals the neural .5*mean5 plus
.5*mean3 objective before decoding/projection. This identity is unit-tested.
Leaf means are decoded to moments; at inference only, shrink benefit/harm
together to the causal maximum disagreement envelope and clamp nested easy
moments. Report the distinction: trees fit the unprojected quadratic objective,
whereas the neural network trains its bounded decoder. This practical estimator
control does not isolate an architecture/optimizer effect perfectly.

## Frozen Comparisons and Readout

Diagnostic primary: source-validation query-balanced signed-score MSE difference
(forest minus frozen source-validation-selected neural checkpoint), averaged by
locality; lower is better. Also compare the final neural checkpoint and its
step-zero prior. No change to the overarching study's accuracy/risk objectives.

After all training, freeze model hashes and source-validation scores. The forest
may be screened using the previously registered finite-completion source rule
(2% selected-reference all/easy upper risk, whole-easy degradation <=2%, positive
worst-case net utility and positive reference denominators). A failing source uses
the existing floor, not an outcome-aware per-row mask. This screen changes no
forest checkpoint, hyperparameter or threshold.

Then freeze all216 causal transfer predictions/actions before reading outcomes.
Compare forest, neural MSE-selected, prior, forest/neural at the same per-query
count, screened forest and floor. Secondary: matched ADE improvement over neural.
Also report normalized signed MSE, ADE/FDE, hard/easy, intervention rate, unknown
selected outcomes, selected-reference all/easy risk and undefined views. Nominal
3000 locality bootstrap replicates and three-seed descriptive means, not a
population/independent significance claim after repeated development exposure.

## Verification and Resource Stop

Require target-algebra, deterministic interrupted resume, causal projection and
train-only preprocessing tests; independently reconstruct validation score and
native readout metrics. Exact first-fit replay and complete frozen readout replay.
Cache/checkpoints remain private. Pilot measures runtime, RSS and compressed size;
conservative node-based remaining storage includes replay and atomic-write copies.
Retain10GiB disk reserve; resource shortage is a blocker, not permission to rename
a smaller run complete. Reuse CREATE only if actual resource evidence justifies it.
